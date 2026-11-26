"""Multi-backend 2D concept preview engine for human confirmation before 3D generation.

Backends:
  1. `diffusers`    - Local GPU (RTX 5090 CUDA bf16 / Apple MPS / CPU) via SDXL-Turbo or FLUX.1-schnell
  2. `openai_image` - Cloud OpenAI-compatible Images API
  3. `studio`       - Zero-GPU instant shaded isometric studio concept board (renders exact AMS colors & mm callouts!)
"""

from __future__ import annotations

import base64
import importlib.util
import math
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spoolgenie.config import PRINTER_PROFILES, SpoolGenieSettings, get_settings
from spoolgenie.engines.cad_engine import build_parametric_model
from spoolgenie.schemas import Concept2DAsset, DesignSpec, EngineRoute

_DIFFUSERS_PIPE_CACHE: dict[str, object] = {}


def _hex_to_rgb(hx: str) -> tuple[int, int, int]:
    hx = hx.lstrip("#")
    if len(hx) != 6:
        return (60, 64, 72)
    return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))


def _load_font(size: int, prefer_cjk: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    cjk_fonts = (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    )
    latin_fonts = (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    order = (cjk_fonts + latin_fonts) if prefer_cjk else (latin_fonts + cjk_fonts)
    for candidate in order:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_shaded_mesh_preview(
    parts: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]],
    width: int = 720,
    height: int = 600,
    azim_deg: float = 38.0,
    elev_deg: float = 26.0,
    bg_rgb: tuple[int, int, int] = (247, 248, 250),
) -> Image.Image:
    """Fast pure-Python painter's-algorithm isometric shaded renderer (zero OpenGL/display needed!)."""
    img = Image.new("RGB", (width, height), bg_rgb)
    draw = ImageDraw.Draw(img)

    if not parts:
        return img

    # Normalize combined vertices to centered unit box
    all_v = np.vstack([v for v, _, _ in parts if len(v) > 0])
    vmin = all_v.min(axis=0)
    vmax = all_v.max(axis=0)
    center = (vmin + vmax) * 0.5
    span = max(float((vmax - vmin).max()), 1e-4)

    az = math.radians(azim_deg)
    el = math.radians(elev_deg)
    # Camera basis for isometric projection (Z-up)
    rz = np.array([[math.cos(az), -math.sin(az), 0.0], [math.sin(az), math.cos(az), 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(el), -math.sin(el)], [0.0, math.sin(el), math.cos(el)]])
    rot = rx @ rz

    # Subtle ground shadow ellipse
    draw.ellipse(
        [width * 0.22, height * 0.72, width * 0.78, height * 0.87],
        fill=(228, 232, 238),
    )

    light_dir = np.array([0.45, -0.55, 0.70], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)

    scale = min(width, height) * 0.62 / span
    cx, cy = width * 0.5, height * 0.52

    # Collect all triangles across parts for back-to-front depth sorting
    tris_2d: list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]] = []

    for verts, faces, base_rgb in parts:
        if len(faces) == 0:
            continue
        v_arr = np.asarray(verts, dtype=np.float64)
        f_arr = np.asarray(faces, dtype=np.int64)

        # Subdivide oversized CAD / planar box faces so painter's algorithm never produces Z-overlap artifacts!
        tri_pts = v_arr[f_arr]
        max_edge = float(
            max(
                np.linalg.norm(tri_pts[:, 1] - tri_pts[:, 0], axis=1).max(),
                np.linalg.norm(tri_pts[:, 2] - tri_pts[:, 1], axis=1).max(),
                np.linalg.norm(tri_pts[:, 0] - tri_pts[:, 2], axis=1).max(),
            )
        )
        if max_edge > span * 0.14 and len(f_arr) < 4000:
            try:
                import trimesh.remesh

                v_arr, f_arr = trimesh.remesh.subdivide_to_size(v_arr, f_arr, max_edge=span * 0.08, max_iter=3)
            except Exception:
                pass

        # Mild subsample only for ultra-dense meshes (>32k faces per part)
        step = max(1, len(f_arr) // 28000)
        f_use = f_arr[::step]
        v_rot = (v_arr - center) @ rot.T
        tri_v = v_rot[f_use]  # (N, 3, 3)

        e1 = tri_v[:, 1] - tri_v[:, 0]
        e2 = tri_v[:, 2] - tri_v[:, 0]
        normals = np.cross(e1, e2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        # Keep non-degenerate, front-facing triangles (camera looks along +Y in rotated coords -> front has normal.y < 0)
        valid = (norms[:, 0] > 1e-8) & (normals[:, 1] <= norms[:, 0] * 0.15)
        if valid.sum() < 4:
            valid = norms[:, 0] > 1e-8
        tri_v = tri_v[valid]
        normals = normals[valid] / norms[valid]

        ndotl = np.clip(normals @ light_dir, -0.35, 1.0)
        shade = np.clip(0.42 + 0.56 * np.abs(ndotl) + 0.14 * np.maximum(normals[:, 2], 0.0), 0.25, 1.06)

        depths = tri_v[:, :, 1].mean(axis=1)
        px = cx + tri_v[:, :, 0] * scale
        py = cy - tri_v[:, :, 2] * scale

        base_arr = np.array(base_rgb, dtype=np.float32)
        for i in range(len(depths)):
            col = tuple(int(c) for c in np.clip(base_arr * shade[i], 0, 255))
            pts = [(float(px[i, 0]), float(py[i, 0])), (float(px[i, 1]), float(py[i, 1])), (float(px[i, 2]), float(py[i, 2]))]
            tris_2d.append((float(depths[i]), pts, col))

    # Draw far-to-near (larger Y is farther away)
    tris_2d.sort(key=lambda item: item[0], reverse=True)
    for _, pts, col in tris_2d:
        draw.polygon(pts, fill=col)

    return img


def render_studio_concept_card(spec: DesignSpec, output_path: Path) -> Concept2DAsset:
    """Instant cross-platform 2D concept confirmation sheet (real CAD preview + AMS swatches + mm callouts)."""
    slot_colors = {f.slot: _hex_to_rgb(f.hex_color) for f in spec.filaments}
    default_rgb = slot_colors.get(1, (43, 45, 48))

    # Build lightweight CAD preview geometry when CAD/hybrid, or stylized organic placeholder silhouette
    preview_parts: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]] = []
    try:
        cad_res = build_parametric_model(spec, step_output_path=None, llm=None)
        for p in cad_res.parts:
            rgb = slot_colors.get(p.slot, default_rgb)
            preview_parts.append((np.asarray(p.mesh.vertices), np.asarray(p.mesh.faces), rgb))
    except Exception:
        pass

    canvas = Image.new("RGB", (1120, 760), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)

    # Left-main hero isometric view + right secondary rear-3/4 view + spec sidebar
    if preview_parts:
        hero = render_shaded_mesh_preview(preview_parts, width=700, height=620, azim_deg=38.0, elev_deg=25.0)
        alt = render_shaded_mesh_preview(preview_parts, width=360, height=290, azim_deg=-132.0, elev_deg=22.0, bg_rgb=(241, 244, 248))
        canvas.paste(hero, (24, 96))
        canvas.paste(alt, (736, 96))
        draw.rectangle([736, 96, 1096, 386], outline=(214, 220, 228), width=2)
    else:
        draw.rounded_rectangle([40, 110, 700, 680], radius=18, fill=(238, 241, 246), outline=(210, 216, 225), width=2)

    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in spec.title)
    f_title = _load_font(26, prefer_cjk=has_cjk)
    f_sub = _load_font(16, prefer_cjk=has_cjk)
    f_sm = _load_font(14, prefer_cjk=has_cjk)

    # Header bar
    draw.rectangle([0, 0, 1120, 76], fill=(24, 28, 36))
    draw.text((28, 16), f"SpoolGenie 2D Concept Gate  |  {spec.title}", fill=(245, 247, 250), font=f_title)
    draw.text(
        (28, 50),
        f"Route: {spec.route.value.upper()}   |   Target: {spec.dimensions.width_mm:.0f} x {spec.dimensions.depth_mm:.0f} x {spec.dimensions.height_mm:.0f} mm",
        fill=(156, 168, 188),
        font=f_sm,
    )

    # Right spec panel
    px0, py0 = 736, 408
    draw.rounded_rectangle([px0, py0, 1096, 720], radius=12, fill=(255, 255, 255), outline=(214, 220, 228), width=2)
    draw.text((px0 + 18, py0 + 14), "Bambu AMS Filament Palette", fill=(28, 34, 44), font=f_sub)

    y = py0 + 48
    for fl in spec.filaments[:4]:
        rgb = _hex_to_rgb(fl.hex_color)
        draw.rounded_rectangle([px0 + 18, y, px0 + 48, y + 30], radius=6, fill=rgb, outline=(180, 185, 192), width=1)
        draw.text((px0 + 58, y + 2), f"Slot {fl.slot}: {fl.name} ({fl.hex_color})", fill=(35, 42, 54), font=f_sm)
        draw.text((px0 + 58, y + 18), f"{fl.material}  •  {fl.role}", fill=(110, 120, 134), font=f_sm)
        y += 44

    printer = PRINTER_PROFILES.get(spec.printer_id, PRINTER_PROFILES["bambu_p1s"])
    draw.line([px0 + 18, y + 8, 1078, y + 8], fill=(226, 230, 236), width=1)
    draw.text((px0 + 18, y + 18), f"Printer: {printer.name}", fill=(35, 42, 54), font=f_sm)
    draw.text(
        (px0 + 18, y + 38),
        f"Dimensions: W {spec.dimensions.width_mm:.0f} × D {spec.dimensions.depth_mm:.0f} × H {spec.dimensions.height_mm:.0f} mm",
        fill=(65, 75, 90),
        font=f_sm,
    )
    draw.text(
        (px0 + 18, y + 58),
        f"Wall: {spec.dimensions.wall_thickness_mm:.1f} mm  |  Clearance: {spec.dimensions.clearance_mm:.2f} mm",
        fill=(65, 75, 90),
        font=f_sm,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return Concept2DAsset(
        backend="studio",
        image_path=str(output_path.resolve()),
        prompt_used=spec.prompt_2d,
        width=canvas.width,
        height=canvas.height,
        notes="Instant shaded multi-view studio concept preview with exact dimensions and Bambu AMS color palette.",
    )


NEGATIVE_PROMPT_3D_PRINT = (
    "text, watermark, signature, logo, label, cluttered background, room interior, table props, "
    "multiple objects, turnaround sheet, split screen, grid, collage, cropped, cut off, "
    "blurry, noisy, lowres, wireframe, dark moody shadows, thin floating wires, broken geometry"
)


def release_diffusers_vram() -> None:
    """Offload cached Diffusers pipelines to CPU and clear CUDA/MPS cache before running 3D generation."""
    import gc

    for pipe in list(_DIFFUSERS_PIPE_CACHE.values()):
        try:
            pipe.to("cpu")
        except Exception:
            pass
    _DIFFUSERS_PIPE_CACHE.clear()
    gc.collect()
    if importlib.util.find_spec("torch"):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_diffusers_concept(
    spec: DesignSpec,
    output_path: Path,
    settings: SpoolGenieSettings,
    lora_path: str | None = None,
) -> Concept2DAsset:
    """Generate a crisp 2D product render locally on RTX 5090 CUDA / Apple MPS / CPU via Diffusers."""
    import torch
    from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

    model_id = settings.diffusers_model
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    cache_key = f"{model_id}:{device}:{dtype}:{lora_path or ''}"
    pipe = _DIFFUSERS_PIPE_CACHE.get(cache_key)
    if pipe is None:
        kwargs = {"torch_dtype": dtype}
        if "xl-base" in model_id.lower() and dtype in (torch.float16, torch.bfloat16):
            kwargs["variant"] = "fp16"
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
        except Exception:
            pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
        if "turbo" not in model_id.lower() and "schnell" not in model_id.lower():
            try:
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config, use_karras_sigmas=True
                )
            except Exception:
                pass
        if lora_path and Path(lora_path).exists():
            pipe.load_lora_weights(lora_path)
        pipe = pipe.to(device)
        _DIFFUSERS_PIPE_CACHE[cache_key] = pipe

    is_turbo = "turbo" in model_id.lower() or "schnell" in model_id.lower()
    steps = 4 if is_turbo else max(16, settings.image_steps)
    guidance = 0.0 if is_turbo else 7.0
    size = 768 if is_turbo else 1024

    gen_kwargs: dict[str, object] = {
        "prompt": spec.prompt_2d,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": size,
        "height": size,
    }
    if not is_turbo:
        gen_kwargs["negative_prompt"] = NEGATIVE_PROMPT_3D_PRINT

    result = pipe(**gen_kwargs)
    img = result.images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return Concept2DAsset(
        backend=f"diffusers:{model_id} ({device})" + (" +LoRA" if lora_path else ""),
        image_path=str(output_path.resolve()),
        prompt_used=spec.prompt_2d,
        width=img.width,
        height=img.height,
        notes=f"Generated locally on {device.upper()} ({dtype}) with {model_id} ({steps} steps).",
    )


def generate_openai_image_concept(
    spec: DesignSpec,
    output_path: Path,
    settings: SpoolGenieSettings,
) -> Concept2DAsset:
    """Generate a 2D concept render via OpenAI-compatible Images API."""
    resp = httpx.post(
        f"{settings.openai_base_url.rstrip('/')}/images/generations",
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        json={
            "model": "gpt-image-1",
            "prompt": spec.prompt_2d,
            "size": "1024x1024",
            "response_format": "b64_json",
        },
        timeout=90.0,
    )
    resp.raise_for_status()
    item = resp.json()["data"][0]
    if "b64_json" in item:
        img_bytes = base64.b64decode(item["b64_json"])
    else:
        img_bytes = httpx.get(item["url"], timeout=30.0).content
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(img_bytes)
    img = Image.open(output_path)
    return Concept2DAsset(
        backend="openai_image",
        image_path=str(output_path.resolve()),
        prompt_used=spec.prompt_2d,
        width=img.width,
        height=img.height,
        notes="Generated via OpenAI-compatible Image API.",
    )


def generate_2d_concept(
    spec: DesignSpec,
    output_path: str | Path,
    settings: SpoolGenieSettings | None = None,
    force_backend: str | None = None,
    lora_path: str | None = None,
) -> Concept2DAsset:
    """Route 2D preview generation to the best available backend for the current hardware & route."""
    settings = settings or get_settings()
    out = Path(output_path)
    choice = (force_backend or settings.image_backend or "auto").lower()

    if choice == "auto":
        # For functional CAD parts, instant shaded studio CAD preview shows exact true geometry + AMS colors immediately!
        # For organic/mesh concepts, prefer local GPU diffusers -> cloud image -> studio fallback.
        if spec.route == EngineRoute.CAD:
            choice = "studio"
        elif importlib.util.find_spec("diffusers") and importlib.util.find_spec("torch"):
            import torch

            if torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                choice = "diffusers"
            elif settings.openai_api_key:
                choice = "openai_image"
            else:
                choice = "studio"
        elif settings.openai_api_key:
            choice = "openai_image"
        else:
            choice = "studio"

    if choice == "diffusers":
        try:
            return generate_diffusers_concept(spec, out, settings, lora_path=lora_path)
        except Exception as exc:
            fallback = render_studio_concept_card(spec, out)
            fallback.notes = f"Diffusers failed ({exc}); fell back to Studio preview."
            return fallback
    elif choice == "openai_image" and settings.openai_api_key:
        try:
            return generate_openai_image_concept(spec, out, settings)
        except Exception as exc:
            fallback = render_studio_concept_card(spec, out)
            fallback.notes = f"OpenAI image failed ({exc}); fell back to Studio preview."
            return fallback

    return render_studio_concept_card(spec, out)


def render_3d_turntable_sheet(
    parts: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]],
    output_path: str | Path,
    title: str,
    subtitle: str = "",
) -> Path:
    """Render a crisp 4-angle (3/4 Front, Front, Side, Rear 3/4) color verification sheet of the ACTUAL 3D model."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    views = [
        ("3/4 Hero View", 35.0, 22.0),
        ("Front View", 0.0, 10.0),
        ("Side View", 90.0, 12.0),
        ("Rear 3/4 View", -140.0, 24.0),
    ]
    canvas = Image.new("RGB", (1120, 820), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)

    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in (title + subtitle))
    f_title = _load_font(24, prefer_cjk=has_cjk)
    f_sub = _load_font(14, prefer_cjk=has_cjk)

    draw.rectangle([0, 0, 1120, 72], fill=(24, 28, 36))
    draw.text((24, 14), f"SpoolGenie 3D Model Turntable  |  {title}", fill=(245, 247, 250), font=f_title)
    if subtitle:
        draw.text((24, 46), subtitle, fill=(156, 168, 188), font=f_sub)

    positions = [(20, 88), (570, 88), (20, 456), (570, 456)]
    for (label, az, el), (x0, y0) in zip(views, positions):
        panel = render_shaded_mesh_preview(parts, width=530, height=348, azim_deg=az, elev_deg=el, bg_rgb=(250, 251, 253))
        canvas.paste(panel, (x0, y0))
        draw.rectangle([x0, y0, x0 + 530, y0 + 348], outline=(214, 220, 228), width=2)
        draw.rectangle([x0 + 8, y0 + 8, x0 + 148, y0 + 32], fill=(24, 28, 36))
        draw.text((x0 + 16, y0 + 12), label, fill=(240, 244, 250), font=f_sub)

    canvas.save(out, format="PNG")
    return out
