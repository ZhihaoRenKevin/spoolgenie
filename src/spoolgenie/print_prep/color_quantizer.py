"""AMS-count-aware color budgeting (CIELAB perceptual merging) + mesh face painting & speckle smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from sklearn.cluster import KMeans

from spoolgenie.config import BAMBU_OFFICIAL_FILAMENTS, BambuOfficialFilament
from spoolgenie.schemas import (
    ColorDecision,
    DesignSpec,
    FilamentMatchRecommendation,
    FilamentSlot,
    MulticolorMode,
)


def hex_to_rgb01(hx: str) -> np.ndarray:
    hx = hx.lstrip("#")
    if len(hx) != 6:
        return np.array([0.2, 0.2, 0.22], dtype=np.float64)
    return np.array([int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)], dtype=np.float64) / 255.0


def rgb01_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Vectorized sRGB [0..1] -> CIE L*a*b* (D65) for perceptually uniform color distance."""
    arr = np.atleast_2d(np.clip(rgb, 0.0, 1.0))
    # Linearize sRGB
    lin = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    # sRGB -> XYZ (D65)
    mat = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = lin @ mat.T
    ref = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
    xyz = xyz / ref
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return np.column_stack([L, a, b])


def delta_e_lab(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(lab1[:, None, :] - lab2[None, :, :], axis=2)


@dataclass
class ColorBudgetResult:
    effective_mode: MulticolorMode
    active_filaments: list[FilamentSlot]
    slot_remap: dict[int, int]  # original 1-based slot -> final 1-based slot
    decisions: list[ColorDecision]


def resolve_ams_color_budget(
    spec: DesignSpec,
    slot_area_weights: dict[int, float] | None = None,
) -> ColorBudgetResult:
    """Decide final filament palette strictly from the user's AMS unit count and multicolor strategy.

    Rules:
      - ams_units >= 1 (AMS mode): max_slots = min(16, ams_units * 4). If design has more colors than
        available AMS slots, merge visually closest / smallest-area colors in CIELAB space and log Delta-E.
      - ams_units == 0 & mode == MONOCHROME: collapse all slots into Slot 1.
      - ams_units == 0 & mode in (AUTO, SPLIT_NO_AMS):
          - If design has >1 color: switch to `SPLIT_NO_AMS` (interlocking glue-free multi-part plates,
            capped at 4 ergonomic color swaps by default unless explicitly kept).
          - If design has 1 color: `MONOCHROME`.
    """
    filaments = list(spec.filaments) or [FilamentSlot(slot=1, name="Matte Charcoal", hex_color="#2B2D30")]
    mode = spec.multicolor_mode

    if mode == MulticolorMode.AUTO:
        if spec.ams_units >= 1:
            effective = MulticolorMode.AMS
        elif len(filaments) > 1:
            effective = MulticolorMode.SPLIT_NO_AMS
        else:
            effective = MulticolorMode.MONOCHROME
    elif mode == MulticolorMode.AMS and spec.ams_units == 0:
        # User set 0 AMS units; gracefully fall back to glue-free split printing for multicolor designs
        effective = MulticolorMode.SPLIT_NO_AMS if len(filaments) > 1 else MulticolorMode.MONOCHROME
    else:
        effective = mode

    # Monochrome collapse
    if effective == MulticolorMode.MONOCHROME:
        primary = filaments[0].model_copy(update={"slot": 1})
        remap = {f.slot: 1 for f in filaments}
        decisions = [
            ColorDecision(
                source_hex=f.hex_color,
                mapped_slot=1,
                mapped_name=primary.name,
                mapped_hex=primary.hex_color,
                delta_e=0.0 if i == 0 else float(round(delta_e_lab(rgb01_to_lab(hex_to_rgb01(f.hex_color)), rgb01_to_lab(hex_to_rgb01(primary.hex_color)))[0, 0], 1)),
                action="kept" if i == 0 else "monochrome_collapsed",
                reason="Primary monochrome filament." if i == 0 else "Collapsed into single-color print.",
            )
            for i, f in enumerate(filaments)
        ]
        return ColorBudgetResult(effective, [primary], remap, decisions)

    max_slots = min(16, spec.ams_units * 4) if effective == MulticolorMode.AMS else 4

    if len(filaments) <= max_slots:
        kept: list[FilamentSlot] = []
        remap = {}
        decisions = []
        for idx, fl in enumerate(filaments, start=1):
            nf = fl.model_copy(update={"slot": idx})
            kept.append(nf)
            remap[fl.slot] = idx
            decisions.append(
                ColorDecision(
                    source_hex=fl.hex_color,
                    mapped_slot=idx,
                    mapped_name=nf.name,
                    mapped_hex=nf.hex_color,
                    delta_e=0.0,
                    action="kept",
                    reason=(
                        f"Fits within {spec.ams_units} AMS unit(s) ({max_slots} slots)."
                        if effective == MulticolorMode.AMS
                        else "Assigned to dedicated glue-free split color plate (No-AMS mode)."
                    ),
                )
            )
        return ColorBudgetResult(effective, kept, remap, decisions)

    # Need to reduce len(filaments) -> max_slots using area-weighted + perceptual CIELAB farthest-point selection
    weights = {f.slot: (slot_area_weights.get(f.slot, 1.0 / (i + 1)) if slot_area_weights else 1.0 / (i + 1)) for i, f in enumerate(filaments)}
    labs = np.vstack([rgb01_to_lab(hex_to_rgb01(f.hex_color)) for f in filaments])

    # Always keep the highest-weight (main body) color first, then greedily pick the highest (weight^0.3 * min_delta_E) color
    sorted_indices = sorted(range(len(filaments)), key=lambda i: weights[filaments[i].slot], reverse=True)
    selected_idx = [sorted_indices[0]]
    while len(selected_idx) < max_slots:
        best_i = None
        best_score = -1.0
        for i in range(len(filaments)):
            if i in selected_idx:
                continue
            d_min = min(float(np.linalg.norm(labs[i] - labs[j])) for j in selected_idx)
            score = d_min * (max(weights[filaments[i].slot], 0.02) ** 0.25)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected_idx.append(best_i)

    selected_idx.sort()  # preserve intuitive slot ordering
    kept_filaments = [
        filaments[i].model_copy(update={"slot": new_slot})
        for new_slot, i in enumerate(selected_idx, start=1)
    ]
    kept_labs = np.vstack([labs[i] for i in selected_idx])

    remap = {}
    decisions = []
    for i, fl in enumerate(filaments):
        if i in selected_idx:
            new_slot = selected_idx.index(i) + 1
            nf = kept_filaments[new_slot - 1]
            remap[fl.slot] = new_slot
            decisions.append(
                ColorDecision(
                    source_hex=fl.hex_color,
                    mapped_slot=new_slot,
                    mapped_name=nf.name,
                    mapped_hex=nf.hex_color,
                    delta_e=0.0,
                    action="kept",
                    reason=f"Selected within {max_slots}-color budget.",
                )
            )
        else:
            dists = np.linalg.norm(kept_labs - labs[i], axis=1)
            nearest = int(np.argmin(dists))
            nf = kept_filaments[nearest]
            de = round(float(dists[nearest]), 1)
            remap[fl.slot] = nf.slot
            decisions.append(
                ColorDecision(
                    source_hex=fl.hex_color,
                    mapped_slot=nf.slot,
                    mapped_name=nf.name,
                    mapped_hex=nf.hex_color,
                    delta_e=de,
                    action="merged",
                    reason=f"Merged into Slot {nf.slot} ({nf.name}) due to {max_slots}-color budget (CIELAB ΔE={de}).",
                )
            )

    return ColorBudgetResult(effective, kept_filaments, remap, decisions)


def sample_mesh_face_rgb(
    mesh: trimesh.Trimesh,
    reference_image_path: str | Path | None = None,
) -> np.ndarray:
    """Return an (F, 3) float64 RGB [0..1] color per triangle face from UV texture, vertex colors, or 2D projection."""
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # 1. Existing face colors
    if hasattr(mesh.visual, "kind") and mesh.visual.kind == "face" and mesh.visual.face_colors is not None:
        fc = np.asarray(mesh.visual.face_colors)[:, :3]
        if len(fc) == n_faces:
            return fc.astype(np.float64) / 255.0

    # 2. Existing vertex colors
    if hasattr(mesh.visual, "kind") and mesh.visual.kind == "vertex" and mesh.visual.vertex_colors is not None:
        vc = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float64) / 255.0
        if len(vc) == len(mesh.vertices):
            return vc[mesh.faces].mean(axis=1)

    # 3. UV texture map
    if hasattr(mesh.visual, "kind") and mesh.visual.kind == "texture":
        try:
            vc_visual = mesh.visual.to_color()
            vc = np.asarray(vc_visual.vertex_colors)[:, :3].astype(np.float64) / 255.0
            if len(vc) == len(mesh.vertices):
                return vc[mesh.faces].mean(axis=1)
        except Exception:
            pass

    # 4. Foreground-clean 2D concept image projection onto untextured Hunyuan3D-2 mesh!
    #    Uses rembg (when installed) or border-color segmentation so white OR studio-gray backdrops never bleed onto edges.
    if reference_image_path and Path(reference_image_path).exists():
        import importlib.util

        from scipy.ndimage import distance_transform_edt

        img_rgba = Image.open(reference_image_path).convert("RGBA")
        if importlib.util.find_spec("rembg"):
            try:
                from rembg import remove

                img_rgba = remove(img_rgba)
            except Exception:
                pass
        arr_rgba = np.asarray(img_rgba, dtype=np.float64)
        rgb = arr_rgba[:, :, :3] / 255.0
        alpha = arr_rgba[:, :, 3] / 255.0

        if (alpha < 0.25).mean() > 0.05:
            fg_mask = alpha > 0.35
        else:
            # Fallback border-color keying (handles both white and neutral studio gray backgrounds)
            corners = np.vstack([rgb[0, :16], rgb[-1, :16], rgb[0, -16:], rgb[-1, -16:]])
            bg_col = np.median(corners, axis=0)
            fg_mask = np.linalg.norm(rgb - bg_col[None, None, :], axis=2) > 0.10

        if fg_mask.sum() > 100:
            ys, xs = np.where(fg_mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            rgb_crop = rgb[y0:y1, x0:x1]
            fg_crop = fg_mask[y0:y1, x0:x1]
            _, indices = distance_transform_edt(~fg_crop, return_indices=True)
            rgb_clean = rgb_crop[indices[0], indices[1]]
        else:
            rgb_clean = rgb

        h_img, w_img, _ = rgb_clean.shape
        centers = mesh.triangles_center
        bmin, bmax = mesh.bounds
        span = np.maximum(bmax - bmin, 1e-4)
        u = np.clip((centers[:, 0] - bmin[0]) / span[0], 0.0, 1.0)
        v = np.clip(1.0 - (centers[:, 2] - bmin[2]) / span[2], 0.0, 1.0)
        px = np.clip((u * (w_img - 1)).astype(int), 0, w_img - 1)
        py = np.clip((v * (h_img - 1)).astype(int), 0, h_img - 1)
        return rgb_clean[py, px]

    # 5. Elevation fallback gradient
    centers = mesh.triangles_center
    zmin, zmax = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
    t = np.clip((centers[:, 2] - zmin) / max(zmax - zmin, 1e-4), 0.0, 1.0)
    return np.column_stack([0.2 + 0.6 * t, 0.25 + 0.35 * t, 0.35 + 0.4 * (1.0 - t)])


def smooth_face_labels(mesh: trimesh.Trimesh, labels: np.ndarray, iterations: int = 8) -> np.ndarray:
    """Spatial majority-vote smoothing + tiny-speckle island absorption to produce clean Bambu AMS color regions."""
    if iterations <= 0 or len(labels) == 0:
        return labels
    adj = mesh.face_adjacency
    if len(adj) == 0:
        return labels

    cur = labels.copy()
    max_label = int(cur.max()) + 1
    n_faces = len(cur)

    for _ in range(iterations):
        votes = np.zeros((n_faces, max_label), dtype=np.int32)
        votes[np.arange(n_faces), cur] += 1
        np.add.at(votes, (adj[:, 0], cur[adj[:, 1]]), 1)
        np.add.at(votes, (adj[:, 1], cur[adj[:, 0]]), 1)
        cur = votes.argmax(axis=1).astype(np.int32)
    return cur


def quantize_mesh_to_ams_slots(
    mesh: trimesh.Trimesh,
    filaments: list[FilamentSlot],
    reference_image_path: str | Path | None = None,
    smooth_iters: int = 8,
) -> tuple[np.ndarray, list[FilamentSlot]]:
    """Assign every triangle face in `mesh` a 1-based AMS slot index matching the active filament budget."""
    if not filaments:
        return np.ones(len(mesh.faces), dtype=np.int32), [FilamentSlot(slot=1)]
    if len(filaments) == 1:
        return np.ones(len(mesh.faces), dtype=np.int32), filaments

    face_rgb = sample_mesh_face_rgb(mesh, reference_image_path=reference_image_path)
    face_lab = rgb01_to_lab(face_rgb)

    pal_lab = np.vstack([rgb01_to_lab(hex_to_rgb01(f.hex_color)) for f in filaments])
    dists = delta_e_lab(face_lab, pal_lab)
    nearest_idx = np.argmin(dists, axis=1).astype(np.int32)

    # If the projected/source image had near-uniform neutral colors (e.g. untextured CAD preview),
    # fall back to K-Means clustering in CIELAB so every selected filament still maps cleanly.
    if len(np.unique(nearest_idx)) == 1 and len(face_lab) >= len(filaments) * 8:
        sample_step = max(1, len(face_lab) // 4000)
        km = KMeans(n_clusters=len(filaments), n_init=5, random_state=42)
        km.fit(face_lab[::sample_step])
        # Match cluster centers to filaments by lightness order
        cluster_order = np.argsort(km.cluster_centers_[:, 0])
        pal_order = np.argsort(pal_lab[:, 0])
        cluster_to_pal = np.zeros(len(filaments), dtype=np.int32)
        for rank in range(len(filaments)):
            cluster_to_pal[cluster_order[rank]] = pal_order[rank]
        nearest_idx = cluster_to_pal[km.predict(face_lab)]

    smoothed = smooth_face_labels(mesh, nearest_idx, iterations=smooth_iters)
    slots = np.array([filaments[idx].slot for idx in smoothed], dtype=np.int32)
    return slots, filaments


def find_closest_bambu_filament(hex_color: str, prefer_name: str = "") -> tuple[BambuOfficialFilament, float]:
    """Return the perceptually closest official Bambu Lab filament and its CIELAB ΔE."""
    target_lab = rgb01_to_lab(hex_to_rgb01(hex_color))[0]
    wants_silk = any(k in prefer_name.lower() for k in ("silk", "gold", "silver", "metal", "chrome", "金", "银"))
    best_item = BAMBU_OFFICIAL_FILAMENTS[0]
    best_score = 1e9
    best_de = 1e9
    for item in BAMBU_OFFICIAL_FILAMENTS:
        item_lab = rgb01_to_lab(hex_to_rgb01(item.hex_color))[0]
        de = float(np.linalg.norm(target_lab - item_lab))
        # Avoid recommending metallic Silk filaments for non-metallic matte/basic canon colors
        penalty = 8.0 if (item.series == "PLA Silk" and not wants_silk) else 0.0
        score = de + penalty
        if score < best_score:
            best_score = score
            best_de = de
            best_item = item
    return best_item, round(best_de, 1)


def match_canon_to_owned_filaments(
    canon_filaments: list[FilamentSlot],
    owned_filaments: list[FilamentSlot] | None = None,
    max_delta_e: float = 22.0,
) -> tuple[list[FilamentSlot], list[FilamentMatchRecommendation]]:
    """Canon-first color matching against user's owned filaments + official Bambu Lab store recommendations.

    Rules:
      1. Preserve the canonical (source-faithful) palette FIRST.
      2. If `owned_filaments` is provided:
         - Compute CIELAB ΔE to every owned filament.
         - If `min_de <= max_delta_e`, bind the slot cleanly to the user's closest owned spool (`owned_match`).
         - If `min_de > max_delta_e` (no close neighbor in inventory!), KEEP the canonical color faithful
           and recommend purchasing the closest official Bambu Lab spool (`purchase_recommended`) with direct
           official store links (`store.bambulab.cn` and `store.bambulab.com`).
      3. If `owned_filaments` is empty, return canonical slots with closest official Bambu store references (`canon_default`).
    """
    owned = list(owned_filaments or [])
    owned_labs = (
        np.vstack([rgb01_to_lab(hex_to_rgb01(f.hex_color)) for f in owned]) if owned else None
    )

    resolved_slots: list[FilamentSlot] = []
    recs: list[FilamentMatchRecommendation] = []

    for idx, cf in enumerate(canon_filaments, start=1):
        bambu_best, _ = find_closest_bambu_filament(cf.hex_color, prefer_name=f"{cf.name} {cf.role}")
        cf_lab = rgb01_to_lab(hex_to_rgb01(cf.hex_color))[0]

        if owned and owned_labs is not None:
            dists = np.linalg.norm(owned_labs - cf_lab[None, :], axis=1)
            best_i = int(np.argmin(dists))
            best_de = round(float(dists[best_i]), 1)
            best_owned = owned[best_i]

            if best_de <= max_delta_e:
                resolved_slots.append(
                    cf.model_copy(
                        update={
                            "slot": idx,
                            "name": best_owned.name,
                            "hex_color": best_owned.hex_color.upper(),
                            "material": best_owned.material,
                        }
                    )
                )
                recs.append(
                    FilamentMatchRecommendation(
                        slot=idx,
                        role=cf.role,
                        canon_name=cf.name,
                        canon_hex=cf.hex_color.upper(),
                        status="owned_match",
                        matched_owned_name=best_owned.name,
                        matched_owned_hex=best_owned.hex_color.upper(),
                        delta_e=best_de,
                        bambu_name=bambu_best.name,
                        bambu_name_zh=bambu_best.name_zh,
                        bambu_hex=bambu_best.hex_color,
                        bambu_series=bambu_best.series,
                        store_url_cn=bambu_best.store_url_cn,
                        store_url_global=bambu_best.store_url_global,
                        reason=f"Matched owned spool '{best_owned.name}' (CIELAB ΔE={best_de} <= {max_delta_e:.0f}).",
                    )
                )
            else:
                # Keep canon color faithful! Recommend official Bambu Lab purchase.
                resolved_slots.append(cf.model_copy(update={"slot": idx}))
                recs.append(
                    FilamentMatchRecommendation(
                        slot=idx,
                        role=cf.role,
                        canon_name=cf.name,
                        canon_hex=cf.hex_color.upper(),
                        status="purchase_recommended",
                        matched_owned_name=best_owned.name,
                        matched_owned_hex=best_owned.hex_color.upper(),
                        delta_e=best_de,
                        bambu_name=bambu_best.name,
                        bambu_name_zh=bambu_best.name_zh,
                        bambu_hex=bambu_best.hex_color,
                        bambu_series=bambu_best.series,
                        store_url_cn=bambu_best.store_url_cn,
                        store_url_global=bambu_best.store_url_global,
                        reason=(
                            f"No close color in your owned spools (nearest '{best_owned.name}', ΔE={best_de} > {max_delta_e:.0f}). "
                            f"Kept canon '{cf.name}' — recommend Bambu Lab {bambu_best.series} {bambu_best.name}."
                        ),
                    )
                )
        else:
            resolved_slots.append(cf.model_copy(update={"slot": idx}))
            recs.append(
                FilamentMatchRecommendation(
                    slot=idx,
                    role=cf.role,
                    canon_name=cf.name,
                    canon_hex=cf.hex_color.upper(),
                    status="canon_default",
                    delta_e=0.0,
                    bambu_name=bambu_best.name,
                    bambu_name_zh=bambu_best.name_zh,
                    bambu_hex=bambu_best.hex_color,
                    bambu_series=bambu_best.series,
                    store_url_cn=bambu_best.store_url_cn,
                    store_url_global=bambu_best.store_url_global,
                    reason=f"Canonical palette matched to Bambu Lab {bambu_best.series} {bambu_best.name}.",
                )
            )

    return resolved_slots, recs
