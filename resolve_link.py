"""How the Sermon-Snip app talks to DaVinci Resolve.

Two transports, one code path (resolve_bridge.py):
  * DIRECT   - Resolve Studio with external scripting enabled: the app calls the bridge itself.
  * SCRIPTS  - any edition that can run Workspace > Scripts: the app installs ONE small script, SermonSnip
               (generated from resolve_bridge.py), and exchanges JSON files in ~/SermonSnip.
"""
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import resolve_bridge as bridge
from tools import HANDOFF_DIR

APP_DIR = Path(__file__).resolve().parent
SNAPSHOT_NAME = "timeline_snapshot.json"
CUT_JOB_NAME = "cut_job.json"
CUT_RESULT_NAME = "cut_result.json"

ENTRY_SYNC = '''

# ---------------------------------------------------------------- entry point (the only script you run)
def _main():
    h = HANDOFF_DIR
    done = []
    job_path, result_path = os.path.join(h, "cut_job.json"), os.path.join(h, "cut_result.json")
    import_path, import_result = os.path.join(h, "import_pending.json"), os.path.join(h, "import_result.json")
    pending = False
    if os.path.exists(job_path):                              # 1. moments waiting to be cut into timelines
        job = read_json(job_path)
        finished = os.path.exists(result_path) and read_json(result_path).get("job_id") == job.get("job_id")
        if not finished:
            pending = True
            report = apply_cut_job(resolve, job)                          # noqa: F821 (provided by Resolve)
            write_json(result_path, report)
            done.append("Created %d timeline(s) for your moments." % len(report["created"]))
            done += ["Problem: " + p for p in report["problems"][:5]]
    if os.path.exists(import_path):                           # 2. finished clips waiting to be imported
        pend = read_json(import_path)
        pending = True
        report = import_package(resolve, pend["manifest"])                # noqa: F821
        report["job_id"] = pend.get("job_id")
        write_json(import_result, report)
        os.remove(import_path)
        done.append("Imported %d of %d finished clip(s) as vertical timelines." % (report["imported"], report["total"]))
        done += ["Problem: " + p for p in report["problems"][:5]]
    if not pending:                                           # 3. nothing waiting: send Sermon-Snip this timeline
        snapshot = take_snapshot(resolve)                                 # noqa: F821
        write_json(os.path.join(h, "timeline_snapshot.json"), snapshot)
        done.append("Sent timeline '%s' (%d tracks) to Sermon-Snip." % (snapshot["timeline"], len(snapshot["tracks"])))
    notify("\\n".join(done) + "\\n\\nSwitch back to Sermon-Snip.")


safe(_main)
'''
GENERATED = {"SermonSnip.py": ENTRY_SYNC}
STATIC = ("SermonSnip_Diagnose.py",)


def build_script(name):
    """Self-contained script text: the bridge source followed by the entry point."""
    return (APP_DIR / "resolve_bridge.py").read_text(encoding="utf-8") + GENERATED[name]


def resolve_scripts_dir():
    """Where Resolve looks for Workspace > Scripts entries (Utility submenu)."""
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", str(Path.home()))) / "Blackmagic Design/DaVinci Resolve/Support/Fusion/Scripts/Utility"
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility"
    return Path.home() / ".local/share/DaVinciResolve/Fusion/Scripts/Utility"


def install_scripts(dest=None):
    dest = Path(dest) if dest else resolve_scripts_dir()
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name in GENERATED:
        (dest / name).write_text(build_script(name), encoding="utf-8")
        written.append(name)
    for name in STATIC:
        src = APP_DIR / "resolve_script" / name
        if src.is_file():
            shutil.copyfile(src, dest / name)
            written.append(name)
    return dest, written


def load_resolve_external():
    """Connect to a running Resolve Studio through its external scripting module -> (resolve|None, reason)."""
    api = os.environ.get("SERMONSNIP_RESOLVE_API")
    if sys.platform == "win32":
        api = api or os.environ.get("RESOLVE_SCRIPT_API") or r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting"
        lib = os.environ.get("RESOLVE_SCRIPT_LIB") or r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll"
    elif sys.platform == "darwin":
        api = api or os.environ.get("RESOLVE_SCRIPT_API") or "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
        lib = os.environ.get("RESOLVE_SCRIPT_LIB") or "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
    else:
        api = api or os.environ.get("RESOLVE_SCRIPT_API") or "/opt/resolve/Developer/Scripting"
        lib = os.environ.get("RESOLVE_SCRIPT_LIB") or "/opt/resolve/libs/Fusion/fusionscript.so"
    os.environ.setdefault("RESOLVE_SCRIPT_API", api)
    os.environ.setdefault("RESOLVE_SCRIPT_LIB", lib)
    modules = os.path.join(api, "Modules")
    if not os.path.isdir(modules):
        return None, "Resolve's scripting module was not found (is DaVinci Resolve installed?)."
    if modules not in sys.path:
        sys.path.append(modules)
    try:
        import DaVinciResolveScript as dvr                          # noqa: N813
        resolve = dvr.scriptapp("Resolve")
    except Exception as exc:
        return None, f"Could not load Resolve's scripting module: {exc}"
    if not resolve:
        return None, ("Resolve did not accept a connection. Direct control needs DaVinci Resolve Studio with "
                      "Preferences > System > General > External scripting using = Local. Use the script route instead.")
    return resolve, "Connected to DaVinci Resolve " + str(bridge.call(resolve, "GetVersionString") or "")


class ResolveLink:
    def __init__(self, handoff=None, external_loader=None):
        self.handoff = Path(handoff) if handoff else Path(HANDOFF_DIR)
        self._loader = external_loader or load_resolve_external
        self.resolve = None
        self.reason = "Not connected."

    # ---- connection
    def connect_direct(self):
        self.resolve, self.reason = self._loader()
        return self.resolve is not None

    @property
    def direct(self):
        return self.resolve is not None

    # ---- snapshot
    @property
    def snapshot_path(self):
        return self.handoff / SNAPSHOT_NAME

    def snapshot_mtime(self):
        try:
            return self.snapshot_path.stat().st_mtime
        except OSError:
            return None

    def read_snapshot(self):
        """Current timeline as a snapshot dict, or None if Resolve hasn't provided one yet."""
        if self.direct:
            return bridge.take_snapshot(self.resolve)
        try:
            return json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # ---- cut moments into timelines
    def submit_cut(self, job):
        """Direct: run now and return the report. Scripts: write the job; returns None (poll cut_result)."""
        job = dict(job, job_id=job.get("job_id") or uuid.uuid4().hex[:8])
        if self.direct:
            return job["job_id"], bridge.apply_cut_job(self.resolve, job)
        bridge.write_json(str(self.handoff / CUT_JOB_NAME), job)
        try:
            (self.handoff / CUT_RESULT_NAME).unlink()
        except OSError:
            pass
        return job["job_id"], None

    def read_cut_result(self, job_id):
        try:
            data = json.loads((self.handoff / CUT_RESULT_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if data.get("job_id") == job_id else None

    def import_package(self, manifest_path):
        """Direct: import now and return the report. Scripts: leave a pending request for the SermonSnip script."""
        if self.direct:
            return bridge.import_package(self.resolve, str(manifest_path))
        self.handoff.mkdir(parents=True, exist_ok=True)
        try:
            (self.handoff / "import_result.json").unlink()
        except OSError:
            pass
        bridge.write_json(str(self.handoff / "import_pending.json"), {"manifest": str(manifest_path), "job_id": uuid.uuid4().hex[:8]})
        return None

    def read_import_result(self):
        try:
            return json.loads((self.handoff / "import_result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # ---- status for the UI
    def status(self):
        snap_time = self.snapshot_mtime()
        scripts_dir = resolve_scripts_dir()
        return {"direct": self.direct, "reason": self.reason, "scripts_dir": str(scripts_dir),
                "scripts_installed": all((scripts_dir / n).is_file() for n in GENERATED),
                "snapshot_age": None if snap_time is None else round(time.time() - snap_time, 1)}
