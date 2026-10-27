"""Local web server for the Sermon-Snip interface. Standard library only.

Security (this process can read files and start programs, so web pages must not be able to drive it):
 * binds to 127.0.0.1 only;
 * every API call needs the per-launch token (sent as X-Token; media URLs carry ?t=);
 * the Host header must be this server (blocks DNS-rebinding);
 * media is served only from the session's work and export folders, never from arbitrary paths.
"""
import json
import mimetypes
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import native
from session import Session, WorkflowError

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_BODY = 1 << 20


class App:
    """Everything the request handler needs; also what tests construct."""

    def __init__(self, session, token=None, pick_file=None, pick_folder=None, open_folder=None, on_quit=None):
        self.session = session
        self.on_quit = on_quit
        self.token = token or secrets.token_urlsafe(16)
        self.pick_file = pick_file or native.pick_file
        self.pick_folder = pick_folder or native.pick_folder
        self.open_folder = open_folder or native.open_folder
        self.port = None

    def quit(self):
        if self.on_quit:
            threading.Timer(0.3, self.on_quit).start()        # reply first, then stop


def _json_ok(**extra):
    return dict({"ok": True}, **extra)


def make_routes(app):
    s = app.session

    def source_file(body):
        path = body.get("path") or app.pick_file()
        if not path:
            return _json_ok(cancelled=True)
        s.load_file(path)
        return _json_ok(path=path)

    def refresh(_):
        return _json_ok(found=s.refresh_from_resolve())

    def connect(_):
        ok = s.link.connect_direct()
        if ok:
            s.refresh_from_resolve()
        return _json_ok(connected=ok, reason=s.link.reason)

    def install(_):
        import resolve_link
        dest, written = resolve_link.install_scripts()
        return _json_ok(folder=str(dest), files=written)

    def tracks(body):
        s.select_tracks([int(i) for i in body.get("audio", [])], [int(i) for i in body.get("video", [])])
        return _json_ok()

    def analyze(body):
        s.analyze(body)
        return _json_ok()

    def rerank(body):
        s.rerank(body.get("target", 60))
        return _json_ok()

    def select(body):
        s.set_selected(body.get("ids", []))
        return _json_ok()

    def preview(body):
        s.request_preview(body["id"])
        return _json_ok()

    def cut(body):
        s.cut_in_resolve(body.get("ids"))
        return _json_ok()

    def reframe(body):
        s.reframe(body.get("ids"), body.get("options", {}))
        return _json_ok()

    def export(body):
        s.export(body.get("ids"), body.get("caption_options", {}), body.get("export_options", {}), body.get("folder") or None)
        return _json_ok()

    def export_all(body):
        s.export_all(body.get("ids"), body.get("reframe_options", {}), body.get("caption_options", {}), body.get("export_options", {}), body.get("folder") or None)
        return _json_ok()

    def adjust(body):
        s.adjust(body["id"], body.get("start"), body.get("end"), bool(body.get("reset")))
        return _json_ok()

    def captions(body):
        s.captions(body.get("ids"), body.get("options", {}))
        return _json_ok()

    def caption_preview(body):
        s.request_caption_preview(body["id"], body.get("options", {}))
        return _json_ok()

    def gemini_test(body):
        from gemini_rank import GeminiError
        try:
            names = s.test_gemini(body.get("key"))
        except GeminiError as exc:
            return {"ok": False, "error": str(exc)}
        return _json_ok(models=names[:12])

    def gemini_key(body):
        s.set_gemini_key(body.get("key", ""))
        return _json_ok()

    def import_presets(body):
        path = body.get("path") or app.pick_file("Choose autosubs-presets.json")
        if path:
            s.locate_tool("autosubs_presets_path", path)
        return _json_ok(path=path, count=s.presets_info.get("user_count"))

    def pick_folder(_):
        path = app.pick_folder("Choose the export folder")
        if path:
            s.custom_export_dir = path
            s.export_dir = Path(path)
            s.bump()
        return _json_ok(path=path)

    def open_export(_):
        folder = s.export_dir or s.planned_export_dir()
        Path(folder).mkdir(parents=True, exist_ok=True)
        app.open_folder(folder)
        return _json_ok()

    def locate(body):
        path = body.get("path") or app.pick_file("Choose the program")
        if path:
            s.locate_tool(body["tool"], path)
        return _json_ok(path=path)

    return {
        "/api/source/file": source_file, "/api/resolve/refresh": refresh, "/api/resolve/connect": connect,
        "/api/resolve/install": install, "/api/tracks": tracks, "/api/analyze": analyze, "/api/rerank": rerank,
        "/api/moments/select": select, "/api/moments/preview": preview, "/api/cut": cut, "/api/reframe": reframe,
        "/api/export": export, "/api/export_all": export_all, "/api/export/folder": pick_folder, "/api/moments/adjust": adjust,
        "/api/captions": captions, "/api/captions/preview": caption_preview, "/api/captions/thumbs": lambda _: (s.request_preset_thumbs(), _json_ok())[1],
        "/api/captions/test": lambda _: (s.run_caption_test(), _json_ok())[1], "/api/captions/presets/import": import_presets,
        "/api/verthor/gpu/check": lambda b: (s.check_verthor_gpu(force=bool(b.get("force"))), _json_ok())[1],
        "/api/verthor/gpu/enable": lambda _: (s.enable_verthor_gpu(), _json_ok())[1],
        "/api/gemini/test": gemini_test, "/api/gemini/key": gemini_key, "/api/export/open": open_export,
        "/api/import": lambda _: (s.import_to_resolve(), _json_ok())[1], "/api/cancel": lambda _: (s.cancel(), _json_ok())[1],
        "/api/gpu/check": lambda _: (s.gpu_check(), _json_ok())[1], "/api/gpu/install": lambda _: (s.gpu_install(), _json_ok())[1],
        "/api/quit": lambda _: (app.quit(), _json_ok())[1], "/api/tools/locate": locate, "/api/tools/refresh": lambda _: (s.refresh_tools(), _json_ok())[1],
    }


def make_handler(app):
    routes = make_routes(app)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "SermonSnip"

        def log_message(self, *args):
            pass

        # ---- helpers
        def _host_ok(self):
            host = (self.headers.get("Host") or "").lower()
            return host in (f"127.0.0.1:{app.port}", f"localhost:{app.port}")

        def _send(self, code, body=b"", ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code, data):
            self._send(code, json.dumps(data).encode("utf-8"))

        def _token_ok(self, query):
            supplied = self.headers.get("X-Token") or (query.get("t") or [""])[0]
            return secrets.compare_digest(supplied, app.token)

        # ---- GET / HEAD
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "bad host"})
            url = urlparse(self.path)
            query = parse_qs(url.query)
            path = unquote(url.path)
            if path == "/":
                html = (WEB_DIR / "index.html").read_text(encoding="utf-8").replace("{{TOKEN}}", app.token)
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            if path in ("/app.css", "/app.js"):
                return self._send(200, (WEB_DIR / path[1:]).read_bytes(), "text/css" if path.endswith("css") else "text/javascript; charset=utf-8")
            if not self._token_ok(query):
                return self._json(401, {"error": "missing or wrong token"})
            if path == "/api/state":
                return self._json(200, app.session.state())
            if path.startswith("/media/"):
                return self._media(path[len("/media/"):])
            return self._json(404, {"error": "not found"})

        def _media(self, rel):
            target = None
            for root in app.session.media_roots():
                candidate = (Path(root) / rel).resolve()
                try:
                    candidate.relative_to(Path(root).resolve())
                except ValueError:
                    continue
                if candidate.is_file():
                    target = candidate
                    break
            if target is None:
                return self._json(404, {"error": "no such file"})
            size = target.stat().st_size
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            start, end, code = 0, size - 1, 200
            rng = self.headers.get("Range")
            if rng:
                m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
                if m and (m.group(1) or m.group(2)):
                    if m.group(1):
                        start = int(m.group(1))
                        end = int(m.group(2)) if m.group(2) else size - 1
                    else:
                        start, end = max(0, size - int(m.group(2))), size - 1
                    if start >= size or start > end:
                        return self._send(416, b"", extra={"Content-Range": f"bytes */{size}"})
                    end, code = min(end, size - 1), 206
            length = end - start + 1
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            try:
                with open(target, "rb") as f:
                    f.seek(start)
                    left = length
                    while left > 0:
                        chunk = f.read(min(1 << 16, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass                                    # the browser closed the stream (normal while scrubbing)

        # ---- POST
        def do_POST(self):
            if not self._host_ok():
                return self._json(403, {"error": "bad host"})
            url = urlparse(self.path)
            if not self._token_ok(parse_qs(url.query)):
                return self._json(401, {"error": "missing or wrong token"})
            handler = routes.get(url.path)
            if not handler:
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    return self._json(413, {"error": "body too large"})
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                if not isinstance(body, dict):
                    raise ValueError("JSON object expected")
            except ValueError as exc:
                return self._json(400, {"error": f"bad request: {exc}"})
            try:
                self._json(200, handler(body))
            except WorkflowError as exc:
                self._json(200, {"ok": False, "error": str(exc)})
            except KeyError as exc:
                self._json(400, {"ok": False, "error": f"missing field {exc}"})
            except Exception as exc:
                self._json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return Handler


def start(session, port=0, **kwargs):
    """Start the server in a background thread. Returns (app, httpd)."""
    app = App(session, **kwargs)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    httpd.daemon_threads = True
    app.port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True, name="http").start()
    return app, httpd


def make_session(**kwargs):
    import resolve_link
    return Session(link=resolve_link.ResolveLink(), **kwargs)
