"""Cross-platform environment & GPU accelerator setup helper.

Usage:
  python scripts/setup_env.py          # Core install (works on any Windows/macOS/Linux CPU)
  python scripts/setup_env.py --gpu    # Auto-detect NVIDIA RTX 50-series (cu128) / 30-40 (cu124) / Apple MPS
  python scripts/setup_env.py --full   # Core + GPU PyTorch + local 2D diffusers + local 3D dependencies
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys


def detect_nvidia_gpu() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="SpoolGenie cross-platform setup helper")
    parser.add_argument("--gpu", action="store_true", help="Install hardware-matched PyTorch (cu128/cu124/mps)")
    parser.add_argument("--local-2d", action="store_true", help="Install local 2D diffusers dependencies")
    parser.add_argument("--local-3d", action="store_true", help="Install local 3D mesh generation dependencies")
    parser.add_argument("--full", action="store_true", help="Install core + GPU PyTorch + local-2d + local-3d + dev")
    args = parser.parse_args()

    use_uv = shutil.which("uv") is not None
    installer = ["uv", "pip", "install"] if use_uv else [sys.executable, "-m", "pip", "install"]

    extras: list[str] = ["dev"]
    if args.local_2d or args.full:
        extras.append("local-2d")
    if args.local_3d or args.full:
        extras.append("local-3d")

    spec = f".[{','.join(extras)}]"
    run([*installer, "-e", spec])

    if args.gpu or args.full:
        sys_name = platform.system()
        gpu_name = detect_nvidia_gpu()
        if gpu_name:
            print(f"Detected NVIDIA GPU: {gpu_name}")
            is_blackwell = any(tag in gpu_name for tag in ("5090", "5080", "5070", "5060", "RTX PRO", "Blackwell"))
            index_url = (
                "https://download.pytorch.org/whl/cu128"
                if is_blackwell
                else "https://download.pytorch.org/whl/cu124"
            )
            print(f"Installing PyTorch from {index_url} ...")
            run([*installer, "--upgrade", "torch", "torchvision", "--index-url", index_url])
        elif sys_name == "Darwin":
            print("Detected macOS — installing standard PyTorch with Apple Silicon MPS support ...")
            run([*installer, "--upgrade", "torch", "torchvision"])
        else:
            print("No NVIDIA GPU detected — installing CPU PyTorch ...")
            run([*installer, "--upgrade", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])

    print("\nSetup finished! Run `spoolgenie doctor` to verify hardware and engines.")


if __name__ == "__main__":
    main()
