# AGENTS.md — Automated Local Deployment & Tool-Use Guide for AI Agents

This file gives **AI coding agents** (Claude Code, OpenAI Codex, Antigravity, Cursor, Windsurf, Cline, OpenClaw, Gemini CLI) deterministic, zero-interaction instructions to **install, deploy, verify, and operate SpoolGenie locally** when a user sends this repository URL.

---

## 1. One-Command Automated Local Install (Recommended)

Run the OS-native installer script (non-interactive, creates `.venv`, installs SpoolGenie, detects hardware):

- **Windows (PowerShell)**:
  ```powershell
  irm https://raw.githubusercontent.com/ZhihaoRenKevin/spoolgenie/main/scripts/install.ps1 | iex
  ```
  Or if already cloned locally:
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts/install.ps1 -Desktop
  ```
  *(Pass `-Gpu` to also install PyTorch CUDA `cu128` for RTX 50-series / `cu124` for RTX 30–40 + local SDXL 2D & Hunyuan3D-2 3D.)*

- **macOS / Linux (bash / zsh)**:
  ```bash
  curl -fsSL https://raw.githubusercontent.com/ZhihaoRenKevin/spoolgenie/main/scripts/install.sh | bash
  ```
  Or with options:
  ```bash
  curl -fsSL https://raw.githubusercontent.com/ZhihaoRenKevin/spoolgenie/main/scripts/install.sh | bash -s -- --desktop --gpu
  ```

---

## 2. Manual Step-by-Step Deployment (If Running Commands Directly)

1. **Clone & enter repo**:
   ```bash
   git clone https://github.com/ZhihaoRenKevin/spoolgenie.git
   cd spoolgenie
   ```
2. **Create Python 3.10–3.12 virtualenv & install core** (prefer `uv` if installed, else `python -m venv`):
   ```bash
   uv venv && uv pip install -e ".[dev,desktop]"
   # fallback: python3 -m venv .venv && .venv/bin/pip install -e ".[dev,desktop]"
   ```
3. **Optional local GPU unlock** (only if user has NVIDIA GPU ≥12 GB VRAM or Apple Silicon ≥16 GB):
   ```bash
   python scripts/setup_env.py --gpu --local-2d --local-3d
   ```
   > **CRITICAL PINNING RULE**: For pure-PyTorch `Hunyuan3D-2` without `nvcc`/MSVC compilation, keep `diffusers>=0.31.0,<0.37.0`, `transformers>=4.46.0,<5.0.0`, `huggingface-hub>=0.26.0,<1.0.0`. On NVIDIA Blackwell (`RTX 5090 / sm_120`), PyTorch **must** be `cu128`.
4. **Verify installation & hardware**:
   ```bash
   spoolgenie doctor
   pytest -q
   ```
5. **Launch local studio**:
   - **Web + Mobile QR Studio (background daemon, works everywhere)**:
     ```bash
     spoolgenie serve --lan --port 8765
     ```
     Report both `http://127.0.0.1:8765` and the Wi-Fi LAN URL to the user.
   - **Native Windows / macOS Desktop Window**:
     ```bash
     spoolgenie desktop
     ```

---

## 3. Connect SpoolGenie as an MCP Server (Direct Tool Calling)

SpoolGenie ships a **zero-dependency stdio MCP server** (`spoolgenie mcp` / `spoolgenie-mcp` / `python -m spoolgenie.mcp_server`).

Inspect exact commands & JSON snippets for the current virtualenv:
```bash
spoolgenie agent-config --json
```

Quick registrations:
- **Claude Code**: `claude mcp add spoolgenie -- <venv-python> -m spoolgenie.mcp_server` (or open repo directly — `.mcp.json` is auto-discovered!)
- **OpenAI Codex CLI**: `codex mcp add spoolgenie -- <venv-python> -m spoolgenie.mcp_server`
- **Cursor / Windsurf / Antigravity / Claude Desktop / OpenClaw**: merge `spoolgenie agent-config --json` → `mcp_json` into the host's MCP config (or run `spoolgenie agent-config --write-cursor`).

### Exposed MCP Tools (7)
1. `spoolgenie_doctor` — Inspect OS, GPU, Ollama models, slicers, and prior counts.
2. `spoolgenie_search_makerworld` — Search live MakerWorld models (`url`, `title`, `thumbnail_url`).
3. `spoolgenie_create_design` — Create/iterate a 3D print design with Bambu AMS count (`0–4`) & glue-free joint type; renders 2D preview PNG.
4. `spoolgenie_build_3d` — Build watertight `.3mf` (Bambu AMS / No-AMS split plates), `.stl`, `.step`, `.glb`, 4-angle turntable PNG, and DFM report.
5. `spoolgenie_colorize_mesh` — Quantize any `.glb/.obj/.stl` into a painted Bambu AMS `.3mf`.
6. `spoolgenie_list_projects` — List projects & exported artifacts.
7. `spoolgenie_lan_info` — Return local + Wi-Fi LAN URLs for phone/browser preview.

---

## 4. Portable Skill

Agents supporting `SKILL.md` (Antigravity, Claude Code, Codex, OpenClaw, Cursor) can copy or symlink `skills/spoolgenie/` directly into their skills folder.

## 5. Development Guardrails for Agents Editing This Repo

- Never expose a character template tab in the Web UI; keep the 171+ subject priors under `src/spoolgenie/knowledge/data/*.json` silent in the backend.
- Keep every `prompt_2d` in `src/spoolgenie/knowledge/data/*.json` strictly `<= 55` words (< 77 SDXL CLIP tokens).
- Run `ruff check src tests desktop scripts` and `pytest -q` before committing changes.
