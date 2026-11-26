# SpoolGenie 线盘神灯 🧞‍♂️🧵

[![CI](https://github.com/ZhihaoRenKevin/spoolgenie/actions/workflows/ci.yml/badge.svg)](https://github.com/ZhihaoRenKevin/spoolgenie/actions/workflows/ci.yml)
[![Release](https://github.com/ZhihaoRenKevin/spoolgenie/actions/workflows/release-desktop.yml/badge.svg)](https://github.com/ZhihaoRenKevin/spoolgenie/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE)

[English](README.md) | 简体中文

> **对着多色耗材盘聊两句许个愿，直接吐出可打印的 `.3MF`。** — 支持 **Windows / macOS 原生桌面端**、**iOS / Android 手机端（PWA 扫码直连 + Capacitor 原生封装）**、**主流 AI Agent 一键自动部署与 MCP 调用**、浏览器与命令行。

用自然语言对话或结合 **MakerWorld** 公开模型参考，先出 **2D 效果图确认配色与毫米级尺寸**，再自动路由至参数化 CAD 或本地 AI 3D 引擎，一键导出可直接拖入 **Bambu Studio / OrcaSlicer** 切片打印的 `.3mf`、`.stl` 与 `.step` 文件。普通无独显电脑开箱即用，配备 **RTX 5090 (`cu128`) / RTX 30–40 (`cu124`) / Apple Silicon MPS** 时自动启用本地 GPU 加速。

---

## 🤖 发给主流 AI Agent，一句话自动安装并本地部署

只需把这个仓库网址发给 **Claude Code、Codex、Antigravity、Cursor、Windsurf、Cline、OpenClaw 或 Gemini CLI**，直接说一句：

> **「帮我按照 `https://github.com/ZhihaoRenKevin/spoolgenie` 里的 `AGENTS.md` 在本机自动安装部署 SpoolGenie，接入它的 MCP 工具并启动界面。」**

仓库已按主流 Agent 标准原生内置全套对接文件：
- **[`AGENTS.md`](AGENTS.md) + [`CLAUDE.md`](CLAUDE.md) + [`llms.txt`](llms.txt)**：Agent 读取网页即可按固定步骤自动拉取、建虚拟环境、识别本机 GPU/CPU、跑 `doctor` 体检并启动服务。
- **跨平台免交互一键部署脚本**：
  - Windows PowerShell：`irm https://raw.githubusercontent.com/ZhihaoRenKevin/spoolgenie/main/scripts/install.ps1 | iex`
  - macOS / Linux：`curl -fsSL https://raw.githubusercontent.com/ZhihaoRenKevin/spoolgenie/main/scripts/install.sh | bash`
- **零依赖原生 MCP Server（[`.mcp.json`](.mcp.json) + `spoolgenie mcp` + `spoolgenie agent-config`）**：对外提供 **7 个标准 MCP 工具**（硬件检测、MakerWorld 实时搜索、2D 概念设计、3D 与拓竹 `.3MF` 导出、网格上色、项目列表、手机扫码局域网地址），Claude Desktop / Claude Code / Codex / Cursor / Antigravity / OpenClaw 均可直接调用。
- **通用 Agent Skill**：[`skills/spoolgenie/SKILL.md`](skills/spoolgenie/SKILL.md) 可直接导入支持 Skill 的任意 Agent。

---

## 核心特性

1. **先看 2D 再出 3D 的对话工作流** — 支持本地 **Ollama** 或 OpenAI 兼容 API；对话栏内实时嵌入 2D 概念图卡片与主画布同步展示，确认颜色、尺寸（mm）与功能后再启动 3D 建模。
2. **双引擎 + 混合功能雕刻路由** —
   - **参数化功能件 (`cad`)**：在 AST 安全沙箱中由 LLM 实时编写 **`build123d`** Python CAD 代码（带自动报错修复），不局限于固定模板，输出精确公差 `.step` / `.stl` / 多实体 `.3mf`。
   - **角色 / 动物 / 植物 / 雕塑 (`mesh`)**：原生纯 PyTorch 版 **Hunyuan3D-2**（Windows / macOS 零 `nvcc` / MSVC 编译），并兼容本地 3D HTTP 服务、云端 API 或零 GPU 浮雕兜底。
   - **实用化角色混合体 (`hybrid`)**：先用 Hunyuan3D-2 生成完整 360° 角色模型，再做水密重建（`Poisson depth=7`）并用 **`manifold3d` 精确毫米级布尔雕刻**（如瓦力背部笔筒、MagSafe 无线充卡槽、手机支架、花盆等，保留角色头部完整特征）。
3. **拓竹 AMS 台数颜色预算（0–4 台 AMS / 1–16 色）** — 在 **CIELAB ΔE** 感知空间自动归并超预算颜色并做杂点平滑，写入原生 Bambu Studio `paint_color` 与 `Metadata/model_settings.config`，减少冲刷废料。
4. **无 AMS 分色拆件 & 超尺寸自动分块（免胶水机械连接）** — 自动通过 `manifold3d` 布尔注入**滑入式燕尾槽 (`dovetail`)**、**弹性开槽卡扣销 (`snap_pin`)** 或**紧配榫卯 (`press_tenon`)**（默认 `0.22 mm` 间隙），自动分盘排版，**0 冲刷废料、免胶水拼装**。
5. **MakerWorld 实时搜索 + 190 项静默先验库 + 本地微调套件** —
   - 左侧内置 **MakerWorld 实时搜索**，支持一键引用、粘贴链接、上传 `.png/.stl/.3mf` 与输入框 **`Ctrl+V` 直接粘贴截图**。
   - 后台静默内置 **171 项精选主题先验**（`src/spoolgenie/knowledge/data/*.json`：漫威/DC、迪士尼/皮克斯、环球/梦工厂/星战/变形金刚、热门动漫游戏、常见动物、植物多肉、交通载具、美食饮品、国潮民俗、节日礼物）与 **19 项 MakerWorld 工程结构标准**（Gridfinity `42 mm`、宜家洞洞板 Skadis `40 mm`、HSW `20 mm`、MagSafe、GoPro、Switch/SD 卡槽、一体打印铰链、机械键盘轴体、VESA/DIN 导轨等），界面清爽不堆砌模板列表，说到相关题材自动生效。
   - 内置本地 RTX GPU **角色 2D LoRA** 与 **MakerWorld CAD QLoRA SFT** 数据集/训练脚本导出命令（`spoolgenie train ...`）。
6. **全平台适配 + 中英文实时切换 (`EN / 中文`)** —
   - **Windows & macOS 原生桌面 App**：基于系统原生 WebView（Windows Edge WebView2 / macOS WKWebView），内嵌后端服务，支持 PyInstaller 打包 `.exe` 与 `.app/.dmg`。
   - **iOS & Android 手机 App**：点击顶部 **`📱 手机`** 生成局域网二维码，同 Wi-Fi 扫码添加到主屏幕即成全屏 PWA 手机 App（底部四栏触控导航），同时提供完整 [`mobile/`](mobile/README.md) Capacitor 6 原生封装工程可打包 `.ipa` / `.apk`。

---

## 快速上手（Windows / macOS / Linux）

### 1. 基础安装（零 GPU、零 Node.js、约 60 秒）

```bash
git clone https://github.com/ZhihaoRenKevin/spoolgenie.git
cd spoolgenie
uv venv && uv pip install -e ".[dev]"
```

### 2. 开启本地 GPU 加速（RTX 5090 `cu128` / RTX 30–40 `cu124` / Apple M 芯片）

```bash
python scripts/setup_env.py --gpu --local-2d --local-3d
spoolgenie doctor
```

### 3. 启动桌面 App 或手机局域网扫码

```bash
# 启动 Windows / macOS 原生桌面端（默认开启局域网，手机可直接扫码连本机 GPU）
pip install -e ".[desktop]"
spoolgenie desktop

# 或用浏览器模式启动并开放局域网手机扫码
spoolgenie serve --lan --port 8765
```

在顶部点击 **`📱 手机`**，用 iPhone Safari（分享 → 添加到主屏幕）或 Android Chrome（菜单 → 安装应用）扫码即可手机全屏操作，所有重型 2D/3D 推理自动在电脑端执行。如需打包独立 `.ipa` / `.apk`，参见 [`mobile/README.md`](mobile/README.md)。

### 4. 打包独立桌面客户端（Windows `.exe` / macOS `.app`）

```bash
pip install -e ".[desktop]"
python scripts/build_desktop.py          # Windows 生成 dist/SpoolGenie/；macOS 生成 .app
python scripts/build_desktop.py --dmg    # macOS 额外打包 .dmg 安装镜像
```

---

## 参与贡献与开源协议

只需提交一行 JSON 即可扩充静默角色/动植物先验库，详见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。本项目采用 MIT 协议开源（[`LICENSE`](LICENSE)）。
