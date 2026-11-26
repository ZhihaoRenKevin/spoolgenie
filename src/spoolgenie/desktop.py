"""Native Windows (Edge WebView2) & macOS (WKWebView) Desktop Application for SpoolGenie.

Embeds the FastAPI + Three.js studio server in-process so users can double-click a single
desktop app (Windows .exe / macOS .app) and simultaneously pair their iPhone/Android phone
over local Wi-Fi via the built-in QR code.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn


def find_available_port(preferred: int = 8765, host: str = "127.0.0.1") -> int:
    """Return `preferred` if free, otherwise ask the OS for an open ephemeral port."""
    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, candidate))
                return int(s.getsockname()[1])
            except OSError:
                continue
    return preferred


def _wait_for_server(port: int, timeout_s: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.35):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _prepare_frozen_workspace() -> None:
    """When launched as a packaged .exe / macOS .app, default workspace to ~/SpoolGenieOutputs."""
    if getattr(sys, "frozen", False) and not os.environ.get("SPOOLGENIE_WORKSPACE_DIR"):
        default_ws = Path.home() / "SpoolGenieOutputs"
        default_ws.mkdir(parents=True, exist_ok=True)
        os.environ["SPOOLGENIE_WORKSPACE_DIR"] = str(default_ws)


def launch_desktop(
    port: int = 8765,
    lan: bool = True,
    width: int = 1420,
    height: int = 890,
    debug: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Start the embedded studio server and open a native OS desktop window."""
    _prepare_frozen_workspace()
    from spoolgenie.web.app import app as fastapi_app
    from spoolgenie.web.app import detect_lan_ips, set_server_runtime

    actual_port = find_available_port(port)
    bind_host = "0.0.0.0" if lan else "127.0.0.1"
    set_server_runtime(bind_host, actual_port)
    os.environ["SPOOLGENIE_BIND_HOST"] = bind_host
    os.environ["SPOOLGENIE_BIND_PORT"] = str(actual_port)

    local_url = f"http://127.0.0.1:{actual_port}"
    lan_ip = detect_lan_ips()[0]
    lan_url = f"http://{lan_ip}:{actual_port}"

    config = uvicorn.Config(
        fastapi_app,
        host=bind_host,
        port=actual_port,
        log_level="warning" if not debug else "info",
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True, name="spoolgenie-uvicorn")
    server_thread.start()

    ready = _wait_for_server(actual_port, timeout_s=8.0)
    info = {
        "ready": ready,
        "local_url": local_url,
        "lan_url": lan_url,
        "bind_host": bind_host,
        "port": actual_port,
    }
    if dry_run:
        server.should_exit = True
        server_thread.join(timeout=3.0)
        return info

    try:
        import webview
    except ImportError:
        print(
            "[SpoolGenie Desktop] pywebview is not installed; opening default browser instead.\n"
            'Install native desktop support with: pip install "spoolgenie[desktop]"'
        )
        print(f"Local URL: {local_url}  |  Mobile LAN: {lan_url}")
        webbrowser.open(local_url)
        try:
            while server_thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            server.should_exit = True
        return info

    webview.create_window(
        title="SpoolGenie",
        url=local_url,
        width=width,
        height=height,
        min_size=(900, 640),
        background_color="#0b0f17",
    )
    try:
        webview.start(debug=debug)
    finally:
        server.should_exit = True
        server_thread.join(timeout=3.0)
    return info


def main() -> None:
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="SpoolGenie Native Desktop App (Windows & macOS)")
    parser.add_argument("--port", type=int, default=8765, help="Embedded server port (default 8765)")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Bind to 127.0.0.1 only instead of allowing same-Wi-Fi mobile QR pairing",
    )
    parser.add_argument("--debug", action="store_true", help="Enable WebView DevTools and verbose logs")
    parser.add_argument("--dry-run", action="store_true", help="Smoke-test server startup and exit cleanly")
    args = parser.parse_args()
    res = launch_desktop(port=args.port, lan=not args.local_only, debug=args.debug, dry_run=args.dry_run)
    if args.dry_run:
        print(res)


if __name__ == "__main__":
    main()
