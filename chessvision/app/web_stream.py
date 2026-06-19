import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2

_lock = threading.Lock()
_jpeg: bytes | None = None

_cmd_lock = threading.Lock()
_cmd_queue: list[str] = []

_games_dir: Path = Path("games")

_HTML = b"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>chessvision</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #111; color: #eee; font-family: sans-serif;
           display: flex; flex-direction: column; align-items: center; padding: 1rem; gap: 0.8rem; }
    h1 { font-size: 1.1rem; letter-spacing: 0.08em; color: #aaa; }
    img { max-width: 100%; max-height: 85vh; border: 2px solid #333; display: block; }
    .controls { display: flex; flex-wrap: wrap; gap: 0.4rem; justify-content: center; }
    button { padding: 0.4em 0.85em; background: #222; color: #ccc;
             border: 1px solid #555; border-radius: 4px; cursor: pointer; font-size: 0.85rem; }
    button:hover { background: #333; border-color: #888; }
    a.games-link { font-size: 0.8rem; color: #888; text-decoration: none; }
    a.games-link:hover { color: #ccc; }
  </style>
</head>
<body>
  <h1>chessvision</h1>
  <img src="/stream" alt="board stream">
  <div class="controls">
    <button onclick="cmd('r')">Record / stop (r)</button>
    <button onclick="cmd('k')">Lock board (k)</button>
    <button onclick="cmd('d')">Toggle detections (d)</button>
    <button onclick="cmd('c')">Corner preview (c)</button>
    <button onclick="cmd('m')">Switch mode (m)</button>
    <button onclick="cmd('v')">Validate (v)</button>
    <button onclick="cmd('o')">Rotate 90\xc2\xb0 (o)</button>
    <button onclick="cmd('f')">Flip board (f)</button>
  </div>
  <a class="games-link" href="/pgn">Download recorded games</a>
  <script>
    function cmd(key) { fetch('/cmd/' + encodeURIComponent(key)).catch(() => {}); }
    document.addEventListener('keydown', function(e) {
      if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
      var key = e.key === 'Escape' ? '\x1b' : (e.key.length === 1 ? e.key : null);
      if (key) { e.preventDefault(); cmd(key); }
    });
  </script>
</body>
</html>
"""

_PGN_LIST_STYLE = b"""
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #111; color: #eee; font-family: sans-serif;
           max-width: 600px; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.1rem; letter-spacing: 0.08em; color: #aaa; margin-bottom: 1rem; }
    ul { list-style: none; display: flex; flex-direction: column; gap: 0.4rem; }
    li a { display: block; padding: 0.4em 0.85em; background: #222; color: #ccc;
            border: 1px solid #555; border-radius: 4px; font-size: 0.85rem;
            text-decoration: none; }
    li a:hover { background: #333; border-color: #888; }
    .empty { color: #666; font-size: 0.9rem; }
    .back { display: inline-block; margin-top: 1rem; font-size: 0.8rem; color: #888; text-decoration: none; }
    .back:hover { color: #ccc; }
"""


def _pgn_list_html() -> bytes:
    files = sorted(_games_dir.glob("*.pgn"), key=lambda p: p.stat().st_mtime, reverse=True)
    if files:
        items = b"".join(
            b"<li><a href='/pgn/" + p.name.encode() + b"' download>" + p.name.encode() + b"</a></li>"
            for p in files
        )
        body = b"<ul>" + items + b"</ul>"
    else:
        body = b"<p class='empty'>No recorded games yet.</p>"
    return (
        b"<!DOCTYPE html><html><head><meta charset='utf-8'><title>chessvision \xe2\x80\x94 games</title>"
        + b"<style>" + _PGN_LIST_STYLE + b"</style></head>"
        + b"<body><h1>Recorded games</h1>" + body
        + b"<a class='back' href='/'>&#8592; back to board</a></body></html>"
    )


def push_frame(frame_bgr) -> None:
    """Encode a BGR frame as JPEG and replace the shared buffer."""
    global _jpeg
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        with _lock:
            _jpeg = buf.tobytes()


def pop_command() -> str | None:
    """Return and remove the oldest pending command key, or None."""
    with _cmd_lock:
        return _cmd_queue.pop(0) if _cmd_queue else None


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass  # silence per-request logs

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_HTML)))
            self.end_headers()
            self.wfile.write(_HTML)

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while True:
                    with _lock:
                        frame = _jpeg
                    if frame is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                    self.wfile.flush()
                    time.sleep(1 / 25)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path.startswith("/cmd/"):
            key = self.path[5:]
            if key:
                with _cmd_lock:
                    _cmd_queue.append(key)
            self.send_response(204)
            self.end_headers()

        elif self.path == "/pgn":
            body = _pgn_list_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/pgn/"):
            name = unquote(self.path[5:])
            # Reject any path traversal attempt
            if "/" in name or "\\" in name or not name.endswith(".pgn"):
                self.send_response(400)
                self.end_headers()
                return
            path = _games_dir / name
            if not path.is_file():
                self.send_response(404)
                self.end_headers()
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-chess-pgn")
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        else:
            self.send_response(404)
            self.end_headers()


def start(host: str = "0.0.0.0", port: int = 8080, games_dir: Path = Path("games")) -> None:
    """Start the MJPEG HTTP server in a background daemon thread."""
    global _games_dir
    _games_dir = games_dir
    server = ThreadingHTTPServer((host, port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[web] streaming at http://{host}:{port}/", flush=True)
