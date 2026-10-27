"""Find and run the external tools Sermon-Snip drives: verthor (reframing) and AutoSubs (captions)."""
import collections
import importlib.util
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
CONFIG_FILE = APP_DIR / "sermon_snip_config.json"
HANDOFF_DIR = Path.home() / "SermonSnip"   # well-known folder the Resolve scripts check first
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class ToolError(RuntimeError):
    """An external tool is missing or failed."""


class Cancelled(Exception):
    """The user pressed Cancel."""


# ---------------------------------------------------------------- config

def load_config():
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- discovery

def _first_existing(paths):
    for p in paths:
        if p and Path(p).is_file():
            return str(p)
    return None


def find_autosubs(cfg=None):
    """Return a command prefix (list) for the AutoSubs CLI, or None."""
    cfg = load_config() if cfg is None else cfg
    local = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", "")
    found = _first_existing([
        cfg.get("autosubs_path"), os.environ.get("SERMONSNIP_AUTOSUBS"),
        shutil.which("autosubs"), shutil.which("AutoSubs"),
        os.path.join(local, "AutoSubs", "AutoSubs.exe") if local else None,        # Windows installer
        os.path.join(program_files, "AutoSubs", "AutoSubs.exe") if program_files else None,
        "/usr/bin/autosubs", "/usr/local/bin/autosubs",                            # Linux / macOS symlink
        "/Applications/AutoSubs.app/Contents/MacOS/autosubs",
        "/Applications/AutoSubs.app/Contents/MacOS/AutoSubs",
    ])
    return [found] if found else None


def find_verthor(cfg=None):
    """Return a command prefix (list) for verthor, or None."""
    cfg = load_config() if cfg is None else cfg
    venv_bin = "Scripts" if os.name == "nt" else "bin"
    exe = "verthor.exe" if os.name == "nt" else "verthor"
    found = _first_existing([
        cfg.get("verthor_path"), os.environ.get("SERMONSNIP_VERTHOR"), shutil.which("verthor"),
        APP_DIR / ".venv-verthor" / venv_bin / exe,
        APP_DIR / ".venv" / venv_bin / exe,
    ])
    if found:
        return [found]
    if importlib.util.find_spec("verthor"):
        return [sys.executable, "-m", "verthor"]
    return None


def tool_report(cfg=None):
    from media import ffmpeg_available
    cfg = load_config() if cfg is None else cfg
    return {"ffmpeg": ffmpeg_available(), "verthor": find_verthor(cfg), "autosubs": find_autosubs(cfg)}


# ---------------------------------------------------------------- running

def run_cancelable(cmd, cancel=None, on_line=None, cwd=None, tail_lines=40):
    """Run cmd, merging stderr into stdout. Returns (returncode, last_lines).

    Polls `cancel` (a threading.Event) every 0.2 s even when the tool prints nothing,
    and terminates the process if it is set.
    """
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", cwd=cwd, creationflags=NO_WINDOW)
    except FileNotFoundError as exc:
        raise ToolError(f"Could not start {cmd[0]!r}: not found.") from exc
    except OSError as exc:
        raise ToolError(f"Could not start {cmd[0]!r}: {exc}") from exc
    lines = queue.Queue()

    def pump():
        for raw in proc.stdout:
            lines.put(raw)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    tail = collections.deque(maxlen=tail_lines)
    finished = False
    try:
        while not finished:
            try:
                raw = lines.get(timeout=0.2)
            except queue.Empty:
                raw = ""
            if raw is None:
                finished = True
            elif raw.strip():
                text = raw.rstrip()
                tail.append(text)
                if on_line:
                    on_line(text)
            if cancel is not None and cancel.is_set():
                proc.terminate()
                try:
                    proc.wait(5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise Cancelled()
        proc.wait()
        return proc.returncode, list(tail)
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass
