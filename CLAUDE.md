# CLAUDE.md — SpoolGenie Guide for Claude Code

Follow [`AGENTS.md`](AGENTS.md) for automated local installation, GPU setup, and verification.

## Quick Local Deploy (When User Pastes This Repo URL)

- **Windows**: `irm https://raw.githubusercontent.com/ZhihaoRenKevin/spoolgenie/main/scripts/install.ps1 | iex`
- **macOS / Linux**: `curl -fsSL https://raw.githubusercontent.com/ZhihaoRenKevin/spoolgenie/main/scripts/install.sh | bash`
- **Verify**: `spoolgenie doctor && pytest -q`
- **Launch Studio**: `spoolgenie serve --lan --port 8765` (or `spoolgenie desktop` for native window)

## MCP Integration

This repository includes `.mcp.json` at the root for automatic Claude Code MCP discovery, plus:
```bash
spoolgenie agent-config
```
which prints the exact `claude mcp add spoolgenie -- <venv-python> -m spoolgenie.mcp_server` registration command.

## Reusable Skill

Portable skill definition: [`skills/spoolgenie/SKILL.md`](skills/spoolgenie/SKILL.md).
