---
name: spoolgenie
description: Install, deploy, and operate SpoolGenie locally to turn natural-language ideas and MakerWorld references into print-ready Bambu Lab AMS multicolor .3mf, No-AMS glue-free split .3mf (dovetail/snap-pin/press-tenon), .stl, and parametric .step files via MCP or CLI. Activate whenever a user asks to install/deploy SpoolGenie or generate 3D-printable models.
---

# SpoolGenie Agent Skill

Use this skill to (1) automatically deploy **SpoolGenie** (`https://github.com/ZhihaoRenKevin/spoolgenie`) on the user's Windows, macOS, or Linux machine, and (2) drive its 2D confirmation gate → 3D dual-engine CAD/mesh → Bambu `.3mf` export pipeline via MCP or CLI.

## 1. Automated Local Install (When Not Yet Installed)

Check if `spoolgenie` is already available (`spoolgenie doctor`). If not, run the native installer:

- **Windows (PowerShell)**:
  ```powershell
  irm https://raw.githubusercontent.com/ZhihaoRenKevin/spoolgenie/main/scripts/install.ps1 | iex
  ```
- **macOS / Linux**:
  ```bash
  curl -fsSL https://raw.githubusercontent.com/ZhihaoRenKevin/spoolgenie/main/scripts/install.sh | bash
  ```
- Pass `-Gpu` (Windows) or `--gpu` (macOS/Linux) only when the machine has an NVIDIA GPU (`>=12 GB` VRAM) or Apple Silicon (`>=16 GB` unified memory) and the user wants local SDXL + Hunyuan3D-2 generation.
- Register SpoolGenie's MCP server into the active agent using `spoolgenie agent-config --json`.

## 2. Standard 3D-Printing Design Workflow

1. **Inspect environment** (`spoolgenie_doctor` / `spoolgenie doctor`) to check GPU, Ollama models, and slicers.
2. **Optional reference search** (`spoolgenie_search_makerworld` / `spoolgenie makerworld "<query>"`) when the user mentions a functional fixture (Gridfinity, Skadis, MagSafe, headphone stand) or wants inspiration.
3. **2D Confirmation Gate** (`spoolgenie_create_design`):
   - Pass exact millimeter dimensions, Bambu printer ID, `ams_units` (`0` = No-AMS glue-free split, `1–4` = `4–16` colors), and `joint_type` (`dovetail`, `snap_pin`, `press_tenon`).
   - Show the generated 2D concept image path (or start the visual Studio via `spoolgenie serve --lan --port 8765`) so the user can verify silhouette, colors, and size before 3D generation.
4. **3D Build & Export** (`spoolgenie_build_3d`):
   - Build watertight `.3mf` (plus `split_plates.3mf` with glue-free joints when `ams_units == 0`), `.stl`, `.step` (CAD route), and `.glb`.
   - Report watertightness, bounding box (`mm`), estimated filament mass (`g`), AMS flush waste (`g`), print time, and clickable output file paths.

## 3. Visual Studio & Mobile Pairing

- Launch browser + Wi-Fi mobile PWA QR pairing: `spoolgenie serve --lan --port 8765`
- Launch native Windows / macOS window: `spoolgenie desktop`
