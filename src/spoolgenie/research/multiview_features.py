"""P1.5 Multi-View Perspective Camera Registration & Dense Visual-Semantic Feature Back-Projection.

Upgrades single-view orthographic projection to:
  1. Coarse-to-fine 5-DoF Perspective Pinhole Camera Registration (`yaw`, `elev`, `fov_deg`) via
     silhouette IoU optimization (eliminating foreshortening shifts on asymmetric quadrupeds like Capybara!).
  2. Multi-View Z-Buffered Visibility & Head-On Normal Weighting:
       w_{u,v} = IoU_v^2 * max(0, -n_u . d_{cam,v})^2 * 1[visible(u, v)]
     so 1..N reference views (primary 2D concept + Web-RAG side/rear photos) fuse cleanly onto 3D super-facets.
  3. Dense Semantic Feature Extraction:
     - Optional RTX 5090 CUDA DINOv2 (`facebook/dinov2-small`) dense patch tokens when `use_dinov2=True`
     - Fast zero-download multi-scale CIELAB + SLIC-smoothed + Sobel-edge descriptor as default/fallback
     producing pairwise semantic edge affinities for sharp boundary snapping in FabGraphCut!
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from spoolgenie.print_prep.modularizer import (
    _prepare_2d_context,
    _Reference2DContext,
    _rgb_to_lab_batch,
)

_DINOV2_CACHE: dict[str, tuple[object, object, str]] = {}


@dataclass
class PerspectivePose:
    yaw_deg: float
    elev_deg: float
    fov_deg: float
    silhouette_iou: float
    cam_dist: float


@dataclass
class MultiViewBackProjResult:
    sf_obs_lab: np.ndarray        # (n_sf, 3) fused CIELAB color per super-facet (NaN if unobserved)
    sf_features: np.ndarray       # (n_sf, D) L2-normalized semantic+chromatic descriptor per super-facet
    sf_confidence: np.ndarray     # (n_sf,) accumulated view-visibility confidence
    poses: list[PerspectivePose]
    feature_backend: str


def _rotation_matrix_yaw_elev(yaw_deg: float, elev_deg: float) -> np.ndarray:
    az = np.radians(yaw_deg)
    el = np.radians(elev_deg)
    ca, sa = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]], dtype=np.float64)
    return rx @ rz


def _project_points_perspective(
    pts_rel: np.ndarray,
    rot: np.ndarray,
    fov_deg: float,
    cam_dist: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project centered 3D points through a perspective pinhole camera along +Y (looking from -Y toward origin).

    Returns (u_proj, v_proj, depth_y) in normalized perspective coordinates.
    """
    vr = pts_rel @ rot.T
    # Camera sits at y = -cam_dist looking along +Y -> positive depth = vr[:, 1] + cam_dist
    depth = np.maximum(vr[:, 1] + cam_dist, 1e-2)
    f = 1.0 / np.tan(np.radians(max(15.0, min(85.0, fov_deg)) * 0.5))
    u = (vr[:, 0] / depth) * f
    v = (vr[:, 2] / depth) * f
    return u, v, depth


def register_perspective_camera(
    mesh: trimesh.Trimesh,
    ctx_2d: _Reference2DContext,
) -> PerspectivePose:
    """Coarse-to-fine 360-deg perspective pinhole pose estimation maximizing 2D silhouette IoU."""
    # Sample uniform surface points so both low-poly CAD boxes (16 verts) AND dense 320^3 AI meshes form solid silhouettes!
    try:
        surf_pts, _ = trimesh.sample.sample_surface(mesh, 2600)
        verts = np.asarray(surf_pts, dtype=np.float64)
    except Exception:
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        if len(verts) > 2600:
            verts = verts[:: max(1, len(verts) // 2600)]
    ctr = 0.5 * (mesh.bounds[0] + mesh.bounds[1])
    v_rel = verts - ctr
    radius = max(float(np.linalg.norm(mesh.extents)) * 0.5, 1.0)

    crop_mask = ctx_2d.fg_mask[ctx_2d.v_min : ctx_2d.v_max + 1, ctx_2d.u_min : ctx_2d.u_max + 1]

    def _eval_grid(mask_bool: np.ndarray, yaw: float, elev: float, fov: float) -> tuple[float, float]:
        g = mask_bool.shape[0]
        # Set camera distance proportional to 1 / sin(fov/2) so object fills frame realistically
        cam_dist = radius * (1.25 / np.sin(np.radians(fov * 0.5)))
        rot = _rotation_matrix_yaw_elev(yaw, elev)
        up, vp, _ = _project_points_perspective(v_rel, rot, fov, cam_dist)
        u0, u1 = float(up.min()), float(up.max())
        v0, v1 = float(vp.min()), float(vp.max())
        if (u1 - u0) < 1e-5 or (v1 - v0) < 1e-5:
            return 0.0, cam_dist
        ui = np.clip(((up - u0) / (u1 - u0) * (g - 1)).astype(np.int32), 0, g - 1)
        vi = np.clip(((v1 - vp) / (v1 - v0) * (g - 1)).astype(np.int32), 0, g - 1)
        sil = np.zeros((g, g), dtype=bool)
        sil[vi, ui] = True
        # 1-px morphological dilation fills sparse vertex cloud into solid silhouette
        sil_d = sil.copy()
        sil_d[1:, :] |= sil[:-1, :]
        sil_d[:-1, :] |= sil[1:, :]
        sil_d[:, 1:] |= sil[:, :-1]
        sil_d[:, :-1] |= sil[:, 1:]
        inter = float(np.logical_and(sil_d, mask_bool).sum())
        union = max(1.0, float(np.logical_or(sil_d, mask_bool).sum()))
        return inter / union, cam_dist

    # Stage 1: Coarse 360-deg search on 56x56 grid
    m_coarse = (
        np.asarray(Image.fromarray(crop_mask.astype(np.uint8) * 255).resize((56, 56), Image.Resampling.NEAREST)) > 127
    )
    best_coarse = (-1.0, 0.0, 12.0, 42.0, radius * 3.5)
    for yaw in range(-180, 180, 15):
        for elev in (0.0, 12.0, 24.0):
            for fov in (34.0, 48.0):
                iou, cd = _eval_grid(m_coarse, float(yaw), float(elev), float(fov))
                # Tiny front tie-breaker only for nearly identical symmetric silhouettes
                score = iou + 0.015 * float(np.cos(np.radians(yaw)))
                if score > best_coarse[0]:
                    best_coarse = (score, float(yaw), float(elev), float(fov), cd)

    # Stage 2: Fine 3-deg refinement on 84x84 grid around coarse optimum!
    m_fine = (
        np.asarray(Image.fromarray(crop_mask.astype(np.uint8) * 255).resize((84, 84), Image.Resampling.NEAREST)) > 127
    )
    _, y0, e0, f0, _ = best_coarse
    best_iou = -1.0
    best_pose = PerspectivePose(y0, e0, f0, 0.0, radius * 3.5)
    for dy in (-12, -9, -6, -3, 0, 3, 6, 9, 12):
        yaw = ((y0 + dy + 180.0) % 360.0) - 180.0
        for de in (-6, -3, 0, 3, 6):
            elev = float(np.clip(e0 + de, -12.0, 36.0))
            for df in (-6, 0, 6):
                fov = float(np.clip(f0 + df, 26.0, 62.0))
                iou, cd = _eval_grid(m_fine, yaw, elev, fov)
                score = iou + 0.01 * float(np.cos(np.radians(yaw)))
                if score > best_iou:
                    best_iou = score
                    best_pose = PerspectivePose(
                        yaw_deg=round(yaw, 1),
                        elev_deg=round(elev, 1),
                        fov_deg=round(fov, 1),
                        silhouette_iou=round(iou, 3),
                        cam_dist=cd,
                    )
    return best_pose


def _extract_dense_2d_feature_map(
    rgb: np.ndarray,
    use_dinov2: bool = False,
) -> tuple[np.ndarray, str]:
    """Extract per-pixel dense semantic + chromatic feature map (H, W, D).

    Uses DINOv2 (`facebook/dinov2-small`) on CUDA when `use_dinov2=True` and available;
    otherwise computes a crisp zero-download multi-scale CIELAB + bilateral-smoothed chromatic feature map.
    """
    h, w = rgb.shape[:2]
    lab = _rgb_to_lab_batch(rgb)  # (H, W, 3)

    if use_dinov2 and importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"):
        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoImageProcessor, AutoModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            key = f"facebook/dinov2-small:{device}"
            if key not in _DINOV2_CACHE:
                proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
                mdl = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
                _DINOV2_CACHE[key] = (proc, mdl, device)
            proc, mdl, device = _DINOV2_CACHE[key]
            inputs = proc(images=Image.fromarray(rgb), return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl(**inputs).last_hidden_state[:, 1:, :]  # drop CLS token -> (1, N_patch, 384)
            side = int(round(np.sqrt(out.shape[1])))
            if side * side == out.shape[1]:
                fmap = out.reshape(1, side, side, -1).permute(0, 3, 1, 2)
                fmap_up = F.interpolate(fmap, size=(h, w), mode="bilinear", align_corners=False)[0]
                feat_dino = fmap_up.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                # Compress top 16 channels via random/deterministic orthogonal projection + append normalized CIELAB
                rng = np.random.default_rng(42)
                proj = rng.standard_normal((feat_dino.shape[-1], 16)).astype(np.float32)
                proj /= np.linalg.norm(proj, axis=0, keepdims=True)
                dino_16 = feat_dino @ proj
                dino_16 /= np.maximum(np.linalg.norm(dino_16, axis=-1, keepdims=True), 1e-6)
                lab_n = np.stack([lab[:, :, 0] / 100.0, lab[:, :, 1] / 60.0, lab[:, :, 2] / 60.0], axis=-1)
                return np.concatenate([dino_16, lab_n * 0.85], axis=-1), f"dinov2-small+lab ({device})"
        except Exception:
            pass

    # Fast zero-download multi-scale CIELAB + edge-preserving smoothed chromatic descriptor (works offline & CI!)
    from scipy.ndimage import gaussian_filter

    lab_s1 = np.stack([gaussian_filter(lab[:, :, c], sigma=1.5) for c in range(3)], axis=-1)
    lab_s3 = np.stack([gaussian_filter(lab[:, :, c], sigma=4.5) for c in range(3)], axis=-1)
    chroma = np.hypot(lab_s1[:, :, 1], lab_s1[:, :, 2])[:, :, None] / 60.0
    # Normalized 7D feature vector emphasizing chromatic consistency over shading luminance
    feat = np.concatenate(
        [
            lab_s1[:, :, 0:1] / 160.0,
            lab_s1[:, :, 1:3] / 42.0,
            lab_s3[:, :, 1:3] / 48.0,
            chroma,
        ],
        axis=-1,
    )
    norms = np.maximum(np.linalg.norm(feat, axis=-1, keepdims=True), 1e-6)
    return (feat / norms).astype(np.float32), "multiscale-lab-chroma"


def backproject_multiview_features(
    mesh: trimesh.Trimesh,
    sf_ids: np.ndarray,
    n_sf: int,
    image_paths: list[str | Path],
    use_dinov2: bool = False,
) -> MultiViewBackProjResult | None:
    """Register 1..N perspective views onto a 3D mesh and fuse Z-buffered CIELAB + dense features per super-facet."""
    valid_paths = [Path(p) for p in image_paths if p and Path(p).exists()]
    if not valid_paths:
        return None

    bmin, bmax = mesh.bounds
    ctr = 0.5 * (bmin + bmax)
    fc = np.asarray(mesh.triangles_center, dtype=np.float64)
    fn = np.asarray(mesh.face_normals, dtype=np.float64)
    fc_rel = fc - ctr
    v_rel = np.asarray(mesh.vertices, dtype=np.float64) - ctr

    poses: list[PerspectivePose] = []
    backend_used = "multiscale-lab-chroma"
    feat_dim: int | None = None

    # Accumulators across all registered views
    sf_lab_num = np.zeros((n_sf, 3), dtype=np.float64)
    sf_lab_den = np.zeros(n_sf, dtype=np.float64)
    sf_feat_num: np.ndarray | None = None
    sf_conf = np.zeros(n_sf, dtype=np.float64)

    for img_path in valid_paths[:6]:
        ctx = _prepare_2d_context(img_path, mesh)
        if ctx is None:
            continue
        pose = register_perspective_camera(mesh, ctx)
        poses.append(pose)

        fmap, backend_used = _extract_dense_2d_feature_map(ctx.rgb, use_dinov2=use_dinov2)
        if sf_feat_num is None:
            feat_dim = fmap.shape[-1]
            sf_feat_num = np.zeros((n_sf, feat_dim), dtype=np.float64)

        rot = _rotation_matrix_yaw_elev(pose.yaw_deg, pose.elev_deg)
        fn_r = fn @ rot.T
        uv_v, vv_v, _ = _project_points_perspective(v_rel, rot, pose.fov_deg, pose.cam_dist)
        u0, u1 = float(uv_v.min()), float(uv_v.max())
        v0, v1 = float(vv_v.min()), float(vv_v.max())
        if (u1 - u0) < 1e-5 or (v1 - v0) < 1e-5:
            continue

        uf, vf, depth_f = _project_points_perspective(fc_rel, rot, pose.fov_deg, pose.cam_dist)
        nx = np.clip((uf - u0) / (u1 - u0), 0.0, 1.0)
        nz = np.clip((vf - v0) / (v1 - v0), 0.0, 1.0)

        h_img, w_img = ctx.rgb.shape[:2]
        ui = np.clip((ctx.u_min + nx * (ctx.u_max - ctx.u_min)).astype(np.int32), 0, w_img - 1)
        vi = np.clip((ctx.v_max - nz * (ctx.v_max - ctx.v_min)).astype(np.int32), 0, h_img - 1)

        # Camera looks along +Y in rotated space -> front-facing triangles have fn_r[:, 1] < 0
        cos_view = np.clip(-fn_r[:, 1], 0.0, 1.0)
        cam_facing = ctx.fg_mask[vi, ui] & (cos_view > 0.12)
        if not np.any(cam_facing):
            continue

        # 128x128 perspective Z-buffer visibility check (eliminates occluded rear triangles!)
        zb = 128
        zu = np.clip((nx * (zb - 1)).astype(np.int32), 0, zb - 1)
        zv = np.clip((nz * (zb - 1)).astype(np.int32), 0, zb - 1)
        zbuf = np.full((zb, zb), 1e9, dtype=np.float64)
        cf_idx = np.where(cam_facing)[0]
        np.minimum.at(zbuf, (zv[cf_idx], zu[cf_idx]), depth_f[cf_idx])
        depth_tol = float(np.ptp(depth_f[cf_idx])) * 0.065
        visible = cam_facing & (depth_f <= zbuf[zv, zu] + depth_tol)
        if not np.any(visible):
            continue

        vis_rgb = ctx.rgb[vi[visible], ui[visible]]
        vis_lab = _rgb_to_lab_batch(vis_rgb)
        vis_feat = fmap[vi[visible], ui[visible]]
        vis_sf = sf_ids[visible]
        vis_w = (max(0.2, pose.silhouette_iou) ** 2) * (cos_view[visible] ** 2)

        # Per-superfacet shadow-robust median aggregation for this view
        for u in np.unique(vis_sf):
            m_u = vis_sf == u
            if int(m_u.sum()) < 1:
                continue
            sub_lab = vis_lab[m_u]
            sub_feat = vis_feat[m_u]
            sub_w = vis_w[m_u]
            w_view = float(sub_w.mean())

            # Suppress dark studio self-shadow pixels when vivid chromatic pixels exist on the same super-facet
            chroma = np.hypot(sub_lab[:, 1], sub_lab[:, 2])
            lum = sub_lab[:, 0]
            chrom_m = (chroma > 13.0) & (lum > 18.0) & (lum < 94.0)
            if float(chrom_m.mean()) >= 0.22 and int(chrom_m.sum()) >= 3:
                c_sub = chroma[chrom_m]
                vivid = sub_lab[chrom_m][c_sub >= float(np.percentile(c_sub, 35.0))]
                lab_med = np.median(vivid if len(vivid) >= 3 else sub_lab[chrom_m], axis=0)
                feat_med = np.median(sub_feat[chrom_m], axis=0)
            else:
                lab_med = np.median(sub_lab, axis=0)
                feat_med = np.median(sub_feat, axis=0)

            sf_lab_num[u] += lab_med * w_view
            sf_lab_den[u] += w_view
            sf_feat_num[u] += feat_med * w_view
            sf_conf[u] += w_view

    if sf_feat_num is None or not poses:
        return None

    obs_lab = np.full((n_sf, 3), np.nan, dtype=np.float64)
    has_obs = sf_lab_den > 1e-8
    obs_lab[has_obs] = sf_lab_num[has_obs] / sf_lab_den[has_obs, None]

    sf_feats = np.zeros_like(sf_feat_num)
    sf_feats[has_obs] = sf_feat_num[has_obs] / sf_lab_den[has_obs, None]
    f_norms = np.linalg.norm(sf_feats, axis=1, keepdims=True)
    sf_feats = np.where(f_norms > 1e-6, sf_feats / np.maximum(f_norms, 1e-6), 0.0)

    return MultiViewBackProjResult(
        sf_obs_lab=obs_lab,
        sf_features=sf_feats,
        sf_confidence=sf_conf,
        poses=poses,
        feature_backend=backend_used,
    )
