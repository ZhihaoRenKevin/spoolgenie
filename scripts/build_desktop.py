"""Cross-platform PyInstaller build script for SpoolGenie Desktop (Windows .exe & macOS .app).

Usage:
    python scripts/build_desktop.py            # builds directory bundle for current OS
    python scripts/build_desktop.py --onefile  # single executable (Windows/Linux)
    python scripts/build_desktop.py --dmg      # also creates macOS .dmg installer (macOS only)
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(onefile: bool = False, make_dmg: bool = False) -> Path:
    os_name = platform.system().lower()
    sep = ";" if os_name == "windows" else ":"
    entry = ROOT / "desktop" / "spoolgenie_desktop.py"
    static_dir = ROOT / "src" / "spoolgenie" / "web" / "static"
    data_dir = ROOT / "src" / "spoolgenie" / "knowledge" / "data"

    app_name = "SpoolGenie"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        app_name,
        "--paths",
        str(ROOT / "src"),
        "--add-data",
        f"{static_dir}{sep}spoolgenie/web/static",
        "--add-data",
        f"{data_dir}{sep}spoolgenie/knowledge/data",
        "--hidden-import",
        "uvicorn.logging",
        "--hidden-import",
        "uvicorn.loops.auto",
        "--hidden-import",
        "uvicorn.protocols.http.auto",
        "--hidden-import",
        "uvicorn.lifespan.on",
        "--hidden-import",
        "segno",
        "--hidden-import",
        "spoolgenie.web.app",
    ]
    if onefile and os_name != "darwin":
        cmd.append("--onefile")
    cmd.append(str(entry))

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    dist_dir = ROOT / "dist"
    out = dist_dir / (f"{app_name}.app" if os_name == "darwin" else app_name)
    if onefile and os_name == "windows":
        out = dist_dir / f"{app_name}.exe"

    if make_dmg and os_name == "darwin" and shutil.which("hdiutil"):
        dmg_path = dist_dir / "SpoolGenie-macOS.dmg"
        if dmg_path.exists():
            dmg_path.unlink()
        subprocess.run(
            [
                "hdiutil",
                "create",
                "-volname",
                "SpoolGenie",
                "-srcfolder",
                str(out),
                "-ov",
                "-format",
                "UDZO",
                str(dmg_path),
            ],
            check=True,
        )
        print(f"Created macOS DMG: {dmg_path}")

    print(f"Desktop build finished: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SpoolGenie native desktop binaries.")
    parser.add_argument("--onefile", action="store_true", help="Build single-file .exe on Windows")
    parser.add_argument("--dmg", action="store_true", help="Wrap macOS .app inside a distributable .dmg")
    args = parser.parse_args()
    build(onefile=args.onefile, make_dmg=args.dmg)


if __name__ == "__main__":
    main()
