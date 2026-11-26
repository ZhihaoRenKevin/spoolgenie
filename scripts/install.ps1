<#
.SYNOPSIS
    Zero-interaction Windows installer for SpoolGenie (ideal for humans & AI Agents).
.EXAMPLE
    irm https://raw.githubusercontent.com/ZhihaoRenKevin/spoolgenie/main/scripts/install.ps1 | iex
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/install.ps1 -Gpu -Launch
#>
param(
    [string]$InstallDir = $(if ($env:SPOOLGENIE_INSTALL_DIR) { $env:SPOOLGENIE_INSTALL_DIR } else { Join-Path $HOME "spoolgenie" }),
    [switch]$Gpu,
    [switch]$NoDesktop,
    [switch]$Launch
)

$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/ZhihaoRenKevin/spoolgenie.git"

Write-Host "`n[SpoolGenie] Starting automated Windows local deployment..." -ForegroundColor Cyan

# Use current directory if already inside a SpoolGenie checkout
if (Test-Path (Join-Path (Get-Location) "src\spoolgenie\__init__.py")) {
    $TargetDir = (Get-Location).Path
    Write-Host "[SpoolGenie] Detected existing checkout: $TargetDir" -ForegroundColor Green
} else {
    $TargetDir = $InstallDir
    if (Test-Path (Join-Path $TargetDir ".git")) {
        Write-Host "[SpoolGenie] Updating existing repo in $TargetDir ..." -ForegroundColor Yellow
        git -C $TargetDir pull --ff-only
    } else {
        Write-Host "[SpoolGenie] Cloning $RepoUrl -> $TargetDir ..." -ForegroundColor Yellow
        git clone $RepoUrl $TargetDir
    }
}

Set-Location $TargetDir

$Extras = if ($NoDesktop) { "dev" } else { "dev,desktop" }
$HasUv = [bool](Get-Command uv -ErrorAction SilentlyContinue)

if ($HasUv) {
    Write-Host "[SpoolGenie] Using uv to create .venv and install SpoolGenie[$Extras]..." -ForegroundColor Cyan
    if (!(Test-Path ".venv")) { uv venv }
    uv pip install -e ".[$Extras]"
} else {
    Write-Host "[SpoolGenie] Using python -m venv to install SpoolGenie[$Extras]..." -ForegroundColor Cyan
    $PyCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } elseif (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { throw "Python 3.10-3.12 is required. Install from https://python.org or install uv." }
    if (!(Test-Path ".venv")) { & $PyCmd -m venv .venv }
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\pip.exe install -e ".[$Extras]"
}

if ($Gpu) {
    Write-Host "[SpoolGenie] Unlocking local NVIDIA CUDA / SDXL / Hunyuan3D-2 acceleration..." -ForegroundColor Cyan
    .\.venv\Scripts\python.exe scripts\setup_env.py --gpu --local-2d --local-3d
}

Write-Host "`n[SpoolGenie] Verifying hardware & environment..." -ForegroundColor Green
.\.venv\Scripts\spoolgenie.exe doctor

Write-Host "`n[SpoolGenie] AI Agent MCP integration snippet:" -ForegroundColor Green
.\.venv\Scripts\spoolgenie.exe agent-config

Write-Host "`n[SpoolGenie] Deployment complete!" -ForegroundColor Green
Write-Host "  Desktop App : $TargetDir\.venv\Scripts\spoolgenie.exe desktop"
Write-Host "  Web + Phone : $TargetDir\.venv\Scripts\spoolgenie.exe serve --lan --port 8765"
Write-Host "  MCP Server  : $TargetDir\.venv\Scripts\spoolgenie-mcp.exe"

if ($Launch) {
    Write-Host "`n[SpoolGenie] Launching Web/PWA Studio with LAN QR pairing on port 8765..." -ForegroundColor Cyan
    .\.venv\Scripts\spoolgenie.exe serve --lan --port 8765
}
