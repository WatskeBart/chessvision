import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

_lock = threading.Lock()
_jpeg: bytes | None = None

_cmd_lock = threading.Lock()
_cmd_queue: list[str] = []

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

        else:
            self.send_response(404)
            self.end_headers()


def start(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the MJPEG HTTP server in a background daemon thread."""
    server = ThreadingHTTPServer((host, port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[web] streaming at http://{host}:{port}/", flush=True)
