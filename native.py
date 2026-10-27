"""Native OS helpers for the web UI: file/folder pickers, open-in-explorer, app-style browser window."""
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_PICK = r'''
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
mode, title = sys.argv[1], sys.argv[2]
if mode == "file":
    p = filedialog.askopenfilename(title=title, filetypes=[("Audio or video", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.mp4 *.mov *.mkv *.mxf *.avi *.webm"), ("All files", "*.*")])
else:
    p = filedialog.askdirectory(title=title)
print(p or "")
'''


def _pick(mode, title):
    """Run the Tk dialog in its own process: Tk misbehaves when driven from a web-server thread."""
    try:
        out = subprocess.run([sys.executable, "-c", _PICK, mode, title], capture_output=True, text=True, timeout=600,
                             creationflags=_NO_WINDOW).stdout.strip()
        return out or None
    except (OSError, subprocess.SubprocessError):
        return None


def pick_file(title="Choose the sermon audio or video"):
    return _pick("file", title)


def pick_folder(title="Choose a folder"):
    return _pick("folder", title)


def open_folder(path):
    path = str(path)
    if sys.platform == "win32":
        os.startfile(path)                                   # noqa: S606 (user-initiated, local path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def open_app_window(url):
    """Open the UI like a desktop app (Edge/Chrome --app mode, no tabs or address bar); else the default browser."""
    candidates = []
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
            if base:
                candidates += [Path(base) / "Microsoft/Edge/Application/msedge.exe", Path(base) / "Google/Chrome/Application/chrome.exe"]
    else:
        candidates += [shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge")]
    for exe in candidates:
        if exe and Path(exe).is_file():
            try:
                subprocess.Popen([str(exe), f"--app={url}", "--window-size=1320,900"], creationflags=_NO_WINDOW)
                return True
            except OSError:
                continue
    return webbrowser.open(url)
