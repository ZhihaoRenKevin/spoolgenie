# SpoolGenie 🧞‍♂️🧵

[![CI](https://github.com/ZhihaoRenKevin/spoolgenie/actions/workflows/ci.yml/badge.svg)](https://github.com/ZhihaoRenKevin/spoolgenie/actions/workflows/ci.yml)
[![Release](https://github.com/ZhihaoRenKevin/spoolgenie/actions/workflows/release-desktop.yml/badge.svg)](https://github.com/ZhihaoRenKevin/spoolgenie/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE)

English | [简体中文](README_CN.md)

> **Wish on a spool, print in reality.** — Cross-platform conversational **2D → 3D → Bambu Lab AMS & Glue-Free `.3MF`** AI Studio for **Windows & macOS Desktop**, **iOS & Android Mobile (PWA + Capacitor)**, Web, and CLI.

SpoolGenie turns natural conversation and live **MakerWorld** references into print-ready **Bambu Studio / OrcaSlicer `.3mf`**, **`.stl`**, and parametric **`.step`** files. It runs on **any CPU laptop** out of the box and automatically unlocks **local NVIDIA CUDA (including RTX 5090 Blackwell `sm_120` / `cu128`)** and **Apple Silicon MPS** when available.

---

## Core Capabilities

1. **Human-in-the-Loop 2D Confirmation Gate** — Discuss ideas, dimensions (`mm`), functions, and filament colors via local **Ollama** or OpenAI-compatible LLMs. Preview and confirm the 2D concept (inline inside chat + full canvas) before generating 3D geometry.
2. **Dual-Engine + Hybrid 3D Routing** —
   - **Parametric Functional Parts (`cad`)**: Synthesizes arbitrary manifold **`build123d`** Python CAD inside an AST sandbox (with automatic self-repair), exporting exact-tolerance `.step`, `.stl`, and multi-body `.3mf`.
   - **Characters, Animals, Plants & Sculptures (`mesh`)**: Local pure-PyTorch **Hunyuan3D-2** (`hy3dgen`, zero `nvcc`/MSVC compilation required), local HTTP 3D servers, optional cloud APIs (Tripo3D), or zero-GPU sculpted relief fallback.
   - **Functional Character Hybrids (`hybrid`)**: Generates a full 360° character mesh via Hunyuan3D-2, reconstructs a watertight manifold (`Poisson depth=7`), and executes exact millimeter **`manifold3d` boolean carving** (e.g., WALL-E rear pen caddy preserving the head, MagSafe charger pockets, phone docks, planters).
3. **Bambu Lab AMS Color Budgeting (`0` to `4` AMS Units / `1–16` Colors)** — Perceptual **CIELAB ΔE** color merging and majority-vote speckle smoothing to cut filament flush waste, writing native Bambu Studio `paint_color` (`4`, `8`, `0C`, `1C`, …) + `Metadata/model_settings.config` + `Metadata/project_settings.config`.
4. **No-AMS Split-by-Color & Oversize Splitting with Glue-Free Mechanical Joints** — Automatically injects watertight `manifold3d` **sliding dovetails (`dovetail`)**, **compliant slotted snap-pins (`snap_pin`)**, or **chamfered press-fit tenons (`press_tenon`)** (`0.22 mm` clearance) onto side-by-side build plates (`0.0 g` AMS poop, zero glue).
5. **Live MakerWorld Search + 170+ Silent Priors + Local Pre-Training** —
   - Search live MakerWorld models directly in the sidebar, paste URLs, upload `.png/.stl/.3mf`, or press `Ctrl+V` to paste screenshots.
   - **171 curated silent subject priors** (`src/spoolgenie/knowledge/data/*.json`) spanning popular movies (Marvel/DC, Disney/Pixar, Universal/DreamWorks/Star Wars/Transformers), anime/games, animals, plants, vehicles, food, mythic/cultural icons, and seasonal decor — automatically triggered behind the scenes without cluttering the UI.
   - **19 built-in MakerWorld engineering CAD standards** (Gridfinity `42 mm`, IKEA Skadis `40 mm`, HSW `20 mm`, MagSafe `56.4 mm`, GoPro/Arca-Swiss, Switch/SD slots, print-in-place hinges, Cherry MX, VESA/DIN rails).
   - Built-in local RTX GPU **2D Character LoRA** & **MakerWorld CAD QLoRA SFT** dataset/script exporters (`spoolgenie train ...`).
6. **Cross-Platform Apps + Bilingual `EN / 中文` UI** —
   - **Windows & macOS Native Desktop App** (`spoolgenie desktop` via Edge WebView2 / WKWebView, packable to `.exe` / `.app` via PyInstaller).
   - **iOS & Android Mobile App** — Install instantly as a touch-optimized **PWA** with bottom tab navigation via the built-in **same-Wi-Fi QR code** (`📱 Mobile`), or build a native `.ipa`/`.apk` from [`mobile/`](mobile/README.md) with Capacitor 6.

---

## Quickstart (Windows, macOS & Linux)

### 1. Core Install (< 60 s, CPU-ready, Zero Node.js)

```bash
git clone https://github.com/ZhihaoRenKevin/spoolgenie.git
cd spoolgenie
uv venv && uv pip install -e ".[dev]"
```

Or using standard `pip`:

```bash
python -m venv .venv
pip install -e ".[dev]"
```

### 2. Optional Local GPU Acceleration (RTX 5090 `cu128` / RTX 30–40 `cu124` / Apple MPS)

```bash
python scripts/setup_env.py --gpu --local-2d --local-3d
spoolgenie doctor
```

### 3. Launch Desktop App, Web Studio, or Mobile Pairing

```bash
# Native Windows / macOS Desktop App (automatically enables Wi-Fi QR pairing for your phone!)
pip install -e ".[desktop]"
spoolgenie desktop

# Or launch Browser Studio with LAN QR pairing for iPhone / Android
spoolgenie serve --lan --port 8765
```

Open **`📱 Mobile`** in the top bar and scan the QR code with iOS Safari or Android Chrome (`Add to Home Screen`) to run SpoolGenie full-screen on your phone while your PC/Mac GPU does the heavy 2D/3D work. See [`mobile/README.md`](mobile/README.md) for native Capacitor `.ipa`/`.apk` builds.

### 4. Package Standalone Windows `.exe` / macOS `.app`

```bash
pip install -e ".[desktop]"
python scripts/build_desktop.py          # Windows dist/SpoolGenie/ or macOS dist/SpoolGenie.app
python scripts/build_desktop.py --dmg    # macOS .dmg installer
```

### 5. CLI Usage & Pre-Training

```bash
# No-AMS glue-free dovetail headphone stand
spoolgenie generate "Desktop headphone stand with cable slot, 210mm tall, matte black and orange" --ams-units 0 --joint dovetail

# Functional character hybrid (1x AMS, 4 colors)
spoolgenie generate "Pixar WALL-E desktop pen holder, 120mm tall, 1 AMS 4 colors" --ams-units 1

# Live MakerWorld search
spoolgenie makerworld "gridfinity desk organizer"

# Expand RAG KB & export local LoRA / QLoRA training kits
spoolgenie kb ingest-makerworld "skadis pegboard hook"
spoolgenie train prepare-character ./my_refs --trigger pf_chibi_print --out outputs/char_lora
spoolgenie train export-cad-sft --out outputs/cad_sft
```

---

## Hardware Compatibility Matrix

| Platform | CAD + Web/Desktop UI + 3MF + Glue-Free Joints | Local LLM / VLM (Ollama) | Local 2D (SDXL) | Local 3D (Hunyuan3D-2) |
| :--- | :--- | :--- | :--- | :--- |
| Any CPU Laptop (Win / Mac / Linux) | Instant | Optional (`qwen2.5:7b` or API) | Studio / Cloud API | Relief Fallback / Cloud API |
| Apple Silicon Mac (M1–M5, 16–32 GB) | Instant | Fast (`qwen2.5:14b` + `qwen2.5vl`) | MPS `fp16` | Pure-PyTorch MPS |
| NVIDIA RTX 3090 / 4090 (24 GB) | Instant | Fast | CUDA `cu124` `bf16` | Pure-PyTorch CUDA |
| NVIDIA RTX 5090 (`sm_120`, 24–32 GB) | Instant | Fast | CUDA `cu128` `bf16` | Pure-PyTorch CUDA (~20 s) |
| iPhone / iPad / Android | Native PWA / Capacitor UI | Pairs over Wi-Fi to PC/Mac | Runs on host GPU | Runs on host GPU |

---

## Contributing & License

Adding new characters, animals, plants, vehicles, or engineering priors takes a 1-line JSON PR — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

MIT License — see [`LICENSE`](LICENSE).
