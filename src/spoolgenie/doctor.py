"""Cross-platform hardware, GPU, Ollama, Python package, and slicer diagnostics."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from spoolgenie.config import SpoolGenieSettings, get_settings


@dataclass
class DoctorReport:
    os_name: str
    arch: str
    python_version: str
    accelerator: str  # "cuda", "mps", "cpu"
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None
    cuda_compute_cap: str | None = None
    torch_version: str | None = None
    torch_cuda_ready: bool = False
    blackwell_warning: str | None = None
    ollama_reachable: bool = False
    ollama_models: list[str] = field(default_factory=list)
    recommended_llm: str | None = None
    recommended_vlm: str | None = None
    packages: dict[str, bool] = field(default_factory=dict)
    slicers: dict[str, str | None] = field(default_factory=dict)
    recommended_2d_backend: str = "studio"
    recommended_3d_organic_backend: str = "relief"
    recommended_3d_cad_backend: str = "build123d"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_slicer_binaries(override: str = "") -> dict[str, str | None]:
    """Locate Bambu Studio and OrcaSlicer across Windows, macOS, and Linux."""
    found: dict[str, str | None] = {"bambu_studio": None, "orca_slicer": None}

    if override and Path(override).exists():
        key = "orca_slicer" if "orca" in override.lower() else "bambu_studio"
        found[key] = str(Path(override).resolve())

    system = platform.system()
    candidates: dict[str, list[Path]] = {"bambu_studio": [], "orca_slicer": []}

    for key, names in (
        ("bambu_studio", ["bambu-studio", "bambu-studio.exe", "BambuStudio"]),
        ("orca_slicer", ["orca-slicer", "orca-slicer.exe", "OrcaSlicer"]),
    ):
        for name in names:
            hit = shutil.which(name)
            if hit:
                found[key] = hit
                break

    if system == "Windows":
        pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        pf86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        candidates["bambu_studio"].extend([
            pf / "Bambu Studio" / "bambu-studio.exe",
            pf86 / "Bambu Studio" / "bambu-studio.exe",
            local / "Programs" / "Bambu Studio" / "bambu-studio.exe",
        ])
        candidates["orca_slicer"].extend([
            pf / "OrcaSlicer" / "orca-slicer.exe",
            pf86 / "OrcaSlicer" / "orca-slicer.exe",
            local / "Programs" / "OrcaSlicer" / "orca-slicer.exe",
        ])
    elif system == "Darwin":
        candidates["bambu_studio"].append(Path("/Applications/BambuStudio.app/Contents/MacOS/BambuStudio"))
        candidates["orca_slicer"].append(Path("/Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer"))
    else:
        candidates["bambu_studio"].extend([Path("/usr/bin/bambu-studio"), Path("/usr/local/bin/bambu-studio")])
        candidates["orca_slicer"].extend([Path("/usr/bin/orca-slicer"), Path("/usr/local/bin/orca-slicer")])

    for key, paths in candidates.items():
        if not found[key]:
            for p in paths:
                if p.exists():
                    found[key] = str(p)
                    break

    return found


def _probe_nvidia_smi() -> tuple[str | None, float | None]:
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        if out:
            line = out.splitlines()[0]
            name, mem_mib = [x.strip() for x in line.split(",")[:2]]
            return name, round(float(mem_mib) / 1024.0, 1)
    except Exception:
        pass
    return None, None


def probe_ollama(base_url: str) -> tuple[bool, list[str]]:
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
        if r.status_code == 200:
            data = r.json()
            models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            return True, models
    except Exception:
        pass
    return False, []


def pick_best_ollama_models(models: list[str]) -> tuple[str | None, str | None]:
    """Select best available chat/code LLM and VLM from local Ollama tags."""
    if not models:
        return None, None

    vlm_prefs = ("qwen2.5vl", "qwen3-vl", "llama3.2-vision", "minicpm-v", "llava", "moondream")
    llm_prefs = ("qwen2.5-coder", "qwen3:32b", "qwen3:14b", "qwen2.5:14b", "qwen2.5:32b", "deepseek-r1", "llama3.1", "qwen2.5:7b", "gemma")

    best_vlm = None
    for pref in vlm_prefs:
        for m in models:
            if pref in m.lower():
                best_vlm = m
                break
        if best_vlm:
            break

    best_llm = None
    for pref in llm_prefs:
        for m in models:
            if pref in m.lower() and "vl" not in m.lower() and "moondream" not in m.lower():
                best_llm = m
                break
        if best_llm:
            break

    if not best_llm:
        non_vlm = [m for m in models if "moondream" not in m.lower()]
        best_llm = non_vlm[0] if non_vlm else models[0]

    return best_llm, best_vlm


def run_doctor(settings: SpoolGenieSettings | None = None) -> DoctorReport:
    settings = settings or get_settings()
    pkgs = [
        "build123d",
        "trimesh",
        "manifold3d",
        "torch",
        "diffusers",
        "transformers",
        "hy3dgen",
        "rembg",
        "pymeshlab",
        "fastapi",
    ]
    pkg_status = {p: importlib.util.find_spec(p) is not None for p in pkgs}

    accelerator = "cpu"
    gpu_name, gpu_vram = _probe_nvidia_smi()
    torch_version = None
    torch_cuda_ready = False
    compute_cap = None
    blackwell_warning = None

    if pkg_status["torch"]:
        try:
            import torch

            torch_version = torch.__version__
            if torch.cuda.is_available():
                torch_cuda_ready = True
                accelerator = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                gpu_vram = round(props.total_memory / (1024**3), 1)
                compute_cap = f"{props.major}.{props.minor}"
                # Check if Blackwell sm_120 is supported by current torch arch list
                if props.major >= 12:
                    arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
                    if arch_list and not any("120" in a or "12.0" in a for a in arch_list):
                        blackwell_warning = (
                            "RTX 50-series (sm_120) detected, but installed PyTorch does not list sm_120. "
                            "Install PyTorch cu128+: python scripts/setup_env.py --gpu"
                        )
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                accelerator = "mps"
                gpu_name = "Apple Silicon (MPS)"
        except Exception:
            pass
    elif gpu_name:
        accelerator = "cpu (NVIDIA GPU present, install PyTorch via scripts/setup_env.py --gpu)"

    ollama_ok, ollama_models = probe_ollama(settings.ollama_base_url)
    rec_llm, rec_vlm = pick_best_ollama_models(ollama_models)
    slicers = find_slicer_binaries(settings.slicer_path)

    # Recommend 2D backend
    if pkg_status["diffusers"] and (torch_cuda_ready or accelerator == "mps"):
        rec_2d = "diffusers"
    elif settings.openai_api_key:
        rec_2d = "openai_image"
    else:
        rec_2d = "studio"

    # Recommend organic 3D backend
    if pkg_status["hy3dgen"] and (torch_cuda_ready or accelerator == "mps"):
        rec_3d = "hunyuan3d"
    elif settings.local_3d_server_url:
        rec_3d = "http_3d"
    elif settings.tripo_api_key:
        rec_3d = "tripo"
    elif settings.meshy_api_key:
        rec_3d = "meshy"
    else:
        rec_3d = "relief"

    return DoctorReport(
        os_name=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        python_version=platform.python_version(),
        accelerator=accelerator,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram,
        cuda_compute_cap=compute_cap,
        torch_version=torch_version,
        torch_cuda_ready=torch_cuda_ready,
        blackwell_warning=blackwell_warning,
        ollama_reachable=ollama_ok,
        ollama_models=ollama_models,
        recommended_llm=rec_llm,
        recommended_vlm=rec_vlm,
        packages=pkg_status,
        slicers=slicers,
        recommended_2d_backend=rec_2d,
        recommended_3d_organic_backend=rec_3d,
        recommended_3d_cad_backend="build123d" if pkg_status["build123d"] else "procedural",
    )
