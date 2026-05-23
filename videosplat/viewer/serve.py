"""
Serve the static viewer directory over HTTP and open a browser tab.
"""

import http.server
import os
import socket
import threading
import time
import webbrowser
from pathlib import Path

from rich.console import Console

console = Console()


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        if self.path.split("?")[0].endswith((".html", ".js")):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()


def serve(viewer_dir: Path, port: int = 8080) -> None:
    os.chdir(viewer_dir)

    for candidate in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", candidate)) != 0:
                port = candidate
                break

    def _open():
        time.sleep(0.5)
        webbrowser.open(f"http://localhost:{port}/index.html")

    threading.Thread(target=_open, daemon=True).start()

    with http.server.HTTPServer(("", port), _Handler) as httpd:
        console.print(
            f"[bold cyan]Viewer running at[/] http://localhost:{port}/index.html\n"
            "Press [bold]Ctrl+C[/] to stop."
        )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            console.print("\n[dim]Viewer stopped.[/]")
