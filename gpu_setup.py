"""NVIDIA runtime discovery, loading and diagnostics for faster-whisper (CTranslate2) on Windows.

Why this module exists
----------------------
CTranslate2 does not link cuBLAS at build time. Its runtime loader looks for ``cublas64_12.dll``
with a plain ``LoadLibraryA`` (normal Windows search order, which includes PATH) and then falls
back to ``%CUDA_PATH%\\bin``. It does NOT use folders registered with Python's
``os.add_dll_directory``. Separately, cuDNN 9 is split into sub-libraries that the bundled
``cudnn64_9.dll`` dispatcher loads by name, and they must be the same cuDNN release.

So we make the libraries findable four independent ways, because Windows picks by loaded-module
name first, then by search order:

1. prepend the folders to ``PATH`` (what ``LoadLibraryA`` consults),
2. point ``CUDA_PATH`` at the cuBLAS folder (CTranslate2's own fallback),
3. ``os.add_dll_directory`` (what Python's ``ctypes`` / extension imports consult),
4. PRELOAD each DLL by absolute path. Once a DLL is in the process, later loads by bare name
   resolve to it without searching at all, so this works even if 1-3 are ignored.

The recommended source of the DLLs is the pinned pip wheels in ``requirements-gpu.txt``, which
avoids a multi-GB CUDA Toolkit install and keeps cuDNN in step with CTranslate2.

Run ``python gpu_setup.py`` for a full diagnosis, ``--selftest`` to also run the real model on
the GPU, or ``--install`` to install the pinned NVIDIA wheels.
"""
import os
import re
import shutil
import site
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

CUBLAS_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll")
# Sub-libraries the cuDNN 9 dispatcher (bundled with CTranslate2) loads by name.
CUDNN_CORE = ("cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn_graph64_9.dll", "cudnn_adv64_9.dll")
CUDNN_PRELOAD_ORDER = (
    "cudnn_graph64_9.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn_adv64_9.dll",
    "cudnn_heuristic64_9.dll", "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll", "cudnn_engines_tensor_ir64_9.dll", "cudnn_ext64_9.dll",
)
# cuDNN release bundled inside the ctranslate2 wheel (read from its cudnn64_9.dll). Sub-libraries
# from a different cuDNN release can trigger CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.
KNOWN_CT2_BUNDLED_CUDNN = {"4.8.3": "9.10.2.21"}
MIN_DRIVER_MAJOR = 525  # CUDA 12.x runtime needs driver >= 525 (minor-version compatibility)

_handles = []        # os.add_dll_directory handles must stay referenced
_preloaded = []      # ctypes handles must stay referenced too
_prepared = None


# ---------------------------------------------------------------- discovery

def _dedupe(paths):
    seen, out = set(), []
    for p in paths:
        try:
            key = str(p.resolve()).lower()
        except OSError:
            key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def site_roots():
    """Every place a pip-installed ``nvidia`` package could live (venv, system, user site, sys.path)."""
    roots = [Path(p) for p in sys.path if p]
    try:
        roots += [Path(p) for p in site.getsitepackages()]
    except Exception:
        pass
    try:
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    return _dedupe([r for r in roots if r.is_dir()])


def _has(folder, names):
    return all((folder / n).is_file() for n in names)


def toolkit_dirs(program_files=None):
    """Installed CUDA Toolkits as [(version_label, bin_dir_or_None)], newest first."""
    pf = Path(program_files or os.environ.get("ProgramFiles", r"C:\Program Files"))
    root = pf / "NVIDIA GPU Computing Toolkit" / "CUDA"
    out = []
    if root.is_dir():
        versions = sorted((v for v in root.glob("v*") if v.is_dir()), reverse=True,
                          key=lambda v: [int(x) for x in re.findall(r"\d+", v.name)] or [0])
        for v in versions:
            # CUDA 13 moved the DLLs to bin\x64; CUDA 12 keeps them in bin. Pick the folder that has them.
            options = [d for d in (v / "bin" / "x64", v / "bin") if d.is_dir()]
            holder = next((d for d in options if any(d.glob("cublas64_*.dll"))), None)
            out.append((v.name, holder or (options[0] if options else None)))
    return out


def cudnn_installer_dirs(program_files=None):
    """The folders NVIDIA's cuDNN 9 installer uses: ...\\CUDNN\\v9.x\\bin and its per-CUDA subfolders."""
    pf = Path(program_files or os.environ.get("ProgramFiles", r"C:\Program Files"))
    root = pf / "NVIDIA" / "CUDNN"
    found = []
    if root.is_dir():
        for v in sorted(root.glob("v*"), reverse=True):
            bin_dir = v / "bin"
            if bin_dir.is_dir():
                found.append((f"cuDNN installer {v.name}", bin_dir))
                found += [(f"cuDNN installer {v.name}", d) for d in sorted(bin_dir.glob("*"), reverse=True) if d.is_dir()]
    return found


def discover(roots=None, env=None, program_files=None):
    """Find folders holding cuBLAS 12 and cuDNN 9, without loading anything.

    Order of preference: pip wheels (version-pinned, match CTranslate2) -> CUDA_PATH -> installed
    CUDA Toolkit -> cuDNN installer -> PATH.
    """
    env = os.environ if env is None else env
    candidates = []                                   # (source, folder)
    for root in (site_roots() if roots is None else [Path(r) for r in roots]):
        for sub in sorted((root / "nvidia").glob("*")) if (root / "nvidia").is_dir() else []:
            for leaf in ("bin", "lib"):
                if (sub / leaf).is_dir():
                    candidates.append(("pip wheel", sub / leaf))
    for key in sorted(k for k in env if k == "CUDA_PATH" or re.fullmatch(r"CUDA_PATH_V12_\d+", k)):
        candidates += [(f"{key}", Path(env[key]) / "bin"), (f"{key}", Path(env[key]) / "bin" / "x64")]
    toolkits = toolkit_dirs(program_files)
    for label, d in toolkits:
        if d:
            candidates.append((f"CUDA Toolkit {label}", d))
    candidates += cudnn_installer_dirs(program_files)
    candidates += [("PATH", Path(p)) for p in env.get("PATH", "").split(os.pathsep) if p]

    result = {"cublas_dir": None, "cublas_source": None, "cudnn_dir": None, "cudnn_source": None,
              "toolkits": [], "searched": _dedupe([d for _, d in candidates])}
    for source, folder in candidates:
        if result["cublas_dir"] is None and folder.is_dir() and _has(folder, CUBLAS_DLLS):
            result["cublas_dir"], result["cublas_source"] = folder, source
        if result["cudnn_dir"] is None and folder.is_dir() and _has(folder, CUDNN_CORE[:3]):
            result["cudnn_dir"], result["cudnn_source"] = folder, source
    for label, d in toolkits:
        result["toolkits"].append({
            "version": label, "bin": d,
            "has_cublas12": bool(d and (d / "cublas64_12.dll").is_file()),
            "has_cublas13": bool(d and (d / "cublas64_13.dll").is_file()),
        })
    return result


# ---------------------------------------------------------------- applying

@dataclass
class PrepareResult:
    found: dict = field(default_factory=dict)
    path_dirs: list = field(default_factory=list)
    preload: list = field(default_factory=list)       # [(dll_name, ok, error_text)]
    cuda_path_before: str = None
    cuda_path_after: str = None


def prepare_cuda_environment(force=False, windows=None, **discover_kwargs):
    """Make cuBLAS/cuDNN loadable by CTranslate2. Safe to call repeatedly. Call BEFORE the first GPU use.

    `windows` exists so the logic can be unit-tested on non-Windows machines; leave it None.
    """
    global _prepared
    windows = (os.name == "nt") if windows is None else windows
    if _prepared is not None and not force:
        return _prepared
    result = PrepareResult(found=discover(**discover_kwargs) if windows else {},
                           cuda_path_before=os.environ.get("CUDA_PATH"))
    if not windows:
        _prepared = result
        return result
    found = result.found
    dirs = [str(found[k]) for k in ("cublas_dir", "cudnn_dir") if found.get(k)]
    dirs = list(dict.fromkeys(dirs))
    result.path_dirs = dirs
    if dirs:
        # 1. PATH: ours first, so a CUDA 13 toolkit earlier on PATH cannot shadow cuBLAS 12.
        lowered = {d.lower() for d in dirs}
        rest = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p and p.lower() not in lowered]
        os.environ["PATH"] = os.pathsep.join(dirs + rest)
        # 2. CTranslate2's own fallback reads %CUDA_PATH%\bin.
        cublas = found.get("cublas_dir")
        if cublas and cublas.name.lower() == "bin":
            os.environ["CUDA_PATH"] = str(cublas.parent)
        result.cuda_path_after = os.environ.get("CUDA_PATH")
        # 3. Python-level DLL search path.
        if hasattr(os, "add_dll_directory"):
            for d in dirs:
                try:
                    _handles.append(os.add_dll_directory(d))
                except OSError:
                    pass
        # 4. Preload by absolute path so bare-name loads resolve to the loaded module.
        result.preload = _preload(found)
    _prepared = result
    return result


def _preload(found):
    import ctypes
    if not hasattr(ctypes, "WinDLL"):      # not Windows (e.g. unit tests simulating it)
        return []
    wanted = []
    if found.get("cublas_dir"):
        wanted += [found["cublas_dir"] / n for n in ("cublasLt64_12.dll", "cublas64_12.dll")]
    if found.get("cudnn_dir"):
        folder = found["cudnn_dir"]
        wanted += [folder / n for n in CUDNN_PRELOAD_ORDER if (folder / n).is_file()]
        # NOT cudnn64_9.dll: the copy bundled with ctranslate2 is the dispatcher that must be used.
    out = []
    for path in wanted:
        try:
            _preloaded.append(ctypes.WinDLL(str(path)))
            out.append((path.name, True, ""))
        except OSError as exc:
            out.append((path.name, False, str(exc)))
    return out


# ---------------------------------------------------------------- system facts

def _run(cmd, timeout=15):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, "", str(exc)


def nvidia_smi():
    """Return {'name','driver','vram_mb'} for the first GPU, or {'error': text}."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        for guess in (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
                      Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"):
            if guess.is_file():
                exe = str(guess)
                break
    if not exe:
        return {"error": "nvidia-smi was not found. The NVIDIA display driver is probably not installed."}
    code, out, err = _run([exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
    if code != 0 or not out:
        return {"error": err or "nvidia-smi failed to talk to the driver."}
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    try:
        return {"name": parts[0], "driver": parts[1], "vram_mb": int(float(parts[2]))}
    except (IndexError, ValueError):
        return {"error": "Unexpected nvidia-smi output: " + out}


def dll_version(path):
    """File version of a Windows DLL as 'a.b.c.d', or None (non-Windows / unreadable)."""
    if os.name != "nt" or not path or not Path(path).is_file():
        return None
    try:
        import ctypes
        from ctypes import wintypes
        ver = ctypes.WinDLL("version")
        size = ver.GetFileVersionInfoSizeW(str(path), None)
        if not size:
            return None
        buf = ctypes.create_string_buffer(size)
        if not ver.GetFileVersionInfoW(str(path), 0, size, buf):
            return None
        ptr, length = ctypes.c_void_p(), wintypes.UINT()
        if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)) or not ptr.value:
            return None
        info = ctypes.cast(ptr, ctypes.POINTER(wintypes.DWORD * 13)).contents
        ms, ls = info[2], info[3]
        return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"
    except Exception:
        return None


def _ct2_dir():
    """Folder of the installed ctranslate2 package, found without importing it."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("ctranslate2")
        return Path(list(spec.submodule_search_locations)[0])
    except Exception:
        return Path("")


def _pkg_version(name):
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        return None


# ---------------------------------------------------------------- diagnosis

@dataclass
class Check:
    name: str
    status: str            # "ok" | "warn" | "fail" | "info"
    detail: str
    fix: str = ""


@dataclass
class Report:
    checks: list = field(default_factory=list)

    def add(self, name, status, detail, fix=""):
        self.checks.append(Check(name, status, detail, fix))

    @property
    def ready(self):
        return not any(c.status == "fail" for c in self.checks)

    def text(self):
        icon = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]", "info": "[info]"}
        lines = []
        for c in self.checks:
            lines.append(f"{icon[c.status]} {c.name}: {c.detail}")
            if c.fix and c.status in ("warn", "fail"):
                lines.append(f"         -> {c.fix}")
        lines.append("")
        lines.append("RESULT: GPU transcription should work." if self.ready else
                     "RESULT: GPU transcription is NOT ready. Fix the [FAIL] items above (top to bottom).")
        return "\n".join(lines)


FIX_INSTALL = ("Click 'Install GPU libraries' in Sermon-Snip, or run INSTALL_GPU_LIBRARIES.bat "
               "(downloads about 1.3 GB of NVIDIA runtime files into this app's own environment).")


def diagnose(selftest=False, model="tiny", prepare=True):
    rep = Report()
    rep.add("Platform", "info", f"{sys.platform}, Python {sys.version.split()[0]} "
            f"({'64' if sys.maxsize > 2**32 else '32'}-bit) at {sys.executable}")
    if sys.maxsize <= 2**32:
        rep.add("Python bitness", "fail", "32-bit Python cannot use CUDA.", "Install 64-bit Python 3.11 or 3.12.")

    smi = nvidia_smi()
    if "error" in smi:
        rep.add("NVIDIA GPU/driver", "fail", smi["error"],
                "Install the current NVIDIA driver (GeForce Experience / nvidia.com/drivers), then restart Windows.")
    else:
        rep.add("NVIDIA GPU/driver", "ok", f"{smi['name']}, driver {smi['driver']}, {smi['vram_mb']} MB VRAM")
        try:
            if int(smi["driver"].split(".")[0]) < MIN_DRIVER_MAJOR:
                rep.add("Driver version", "fail", f"Driver {smi['driver']} is too old for CUDA 12 libraries "
                        f"(needs {MIN_DRIVER_MAJOR}+).", "Update the NVIDIA driver.")
        except ValueError:
            pass
        if smi["vram_mb"] < 5000:
            rep.add("VRAM", "warn", f"{smi['vram_mb']} MB is tight for large-v3 with long chunks.",
                    "Use the 'medium' or 'distil-large-v3' model if you see out-of-memory errors.")

    ct2_ver, fw_ver = _pkg_version("ctranslate2"), _pkg_version("faster-whisper")
    if not ct2_ver:
        rep.add("CTranslate2", "fail", "Not installed.", "Run INSTALL_AND_RUN.bat (it installs requirements.txt).")
        return rep
    rep.add("CTranslate2 / faster-whisper", "ok", f"ctranslate2 {ct2_ver}, faster-whisper {fw_ver or 'not installed'}")
    if not fw_ver:
        rep.add("faster-whisper", "fail", "Not installed.", "Run INSTALL_AND_RUN.bat.")

    res = prepare_cuda_environment(force=True) if prepare else (_prepared or PrepareResult())
    found = res.found or discover()

    # --- cuBLAS
    if found.get("cublas_dir"):
        rep.add("cuBLAS 12 (cublas64_12.dll)", "ok", f"{found['cublas_dir']}  [{found['cublas_source']}]")
    else:
        extra = ""
        v13_only = [t for t in found.get("toolkits", []) if t["has_cublas13"] and not t["has_cublas12"]]
        if v13_only:
            extra = (f" A CUDA Toolkit {v13_only[0]['version']} IS installed, but it contains cublas64_13.dll. "
                     "That is the wrong major version: CTranslate2 needs the CUDA 12 file.")
        rep.add("cuBLAS 12 (cublas64_12.dll)", "fail", "Not found anywhere Sermon-Snip looked." + extra, FIX_INSTALL)

    # --- cuDNN
    if found.get("cudnn_dir"):
        rep.add("cuDNN 9 sub-libraries", "ok", f"{found['cudnn_dir']}  [{found['cudnn_source']}]")
        wheel = _pkg_version("nvidia-cudnn-cu12")
        bundled = dll_version(_ct2_dir() / "cudnn64_9.dll") or KNOWN_CT2_BUNDLED_CUDNN.get(ct2_ver)
        installed = wheel or dll_version(found["cudnn_dir"] / "cudnn_ops64_9.dll")
        if bundled and installed and installed.split(".")[:3] != bundled.split(".")[:3]:
            rep.add("cuDNN version match", "warn",
                    f"CTranslate2 {ct2_ver} bundles cuDNN {bundled}, but the sub-libraries found are {installed}. "
                    "A mismatch can cause CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH at the first transcription.",
                    "Install the pinned wheels: " + FIX_INSTALL)
        elif bundled and installed:
            rep.add("cuDNN version match", "ok", f"{installed} matches CTranslate2's bundled {bundled}")
    else:
        rep.add("cuDNN 9 sub-libraries", "fail", "cudnn_ops64_9.dll / cudnn_cnn64_9.dll / cudnn_graph64_9.dll not found.",
                FIX_INSTALL)

    # --- can Windows actually load them?
    if os.name == "nt":
        bad = [(n, e) for n, ok, e in res.preload if not ok]
        if res.preload and not bad:
            rep.add("Loading the DLLs", "ok", f"{len(res.preload)} libraries loaded into the process")
        for n, e in bad:
            rep.add(f"Loading {n}", "fail", e or "load failed",
                    "Usually a missing dependency: install the Microsoft Visual C++ 2015-2022 x64 redistributable "
                    "(aka.ms/vs/17/release/vc_redist.x64.exe) and make sure the NVIDIA driver is current.")
    else:
        rep.add("Loading the DLLs", "info", "Windows-only check skipped on this platform.")

    # --- does CTranslate2 see the device?
    try:
        import ctranslate2
        count = ctranslate2.get_cuda_device_count()
        if count > 0:
            types = sorted(ctranslate2.get_supported_compute_types("cuda"))
            rep.add("CTranslate2 CUDA devices", "ok", f"{count} device(s); compute types: {', '.join(types)}")
        else:
            rep.add("CTranslate2 CUDA devices", "fail", "CTranslate2 reports 0 CUDA devices.",
                    "Update the NVIDIA driver and restart Windows; confirm 'nvidia-smi' works in a terminal.")
    except Exception as exc:
        rep.add("CTranslate2 CUDA devices", "fail", f"{type(exc).__name__}: {exc}")

    if selftest and rep.ready:
        rep.checks.append(_selftest(model))
    return rep


def _selftest(model):
    """Load a real model on the GPU and transcribe 3 s of silence. This exercises cuBLAS AND cuDNN."""
    try:
        import numpy as np
        from faster_whisper import WhisperModel
        m = WhisperModel(model, device="cuda", compute_type="float16")
        segments, _ = m.transcribe(np.zeros(16000 * 3, dtype=np.float32), language="en")
        list(segments)  # transcription is lazy: force it
        return Check("GPU self-test", "ok", f"'{model}' ran on the GPU (float16).")
    except Exception as exc:
        return Check("GPU self-test", "fail", f"{type(exc).__name__}: {exc}",
                     "Read the message above: 'cublas' = cuBLAS problem, 'cudnn' = cuDNN problem, "
                     "'out of memory' = choose a smaller model.")


# ---------------------------------------------------------------- installing

def install_gpu_runtime(on_line=None, cancel=None):
    """pip-install the pinned NVIDIA runtime wheels into the running interpreter's environment."""
    from tools import run_cancelable
    req = APP_DIR / "requirements-gpu.txt"
    code, tail = run_cancelable([sys.executable, "-m", "pip", "install", "-r", str(req)],
                                cancel=cancel, on_line=on_line, tail_lines=30)
    if code != 0:
        raise RuntimeError("pip could not install the NVIDIA libraries:\n" + "\n".join(tail[-8:]))
    return True


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--install" in argv:
        print("Installing pinned NVIDIA runtime wheels (about 1.3 GB)...")
        install_gpu_runtime(on_line=print)
        print("Done. Restart Sermon-Snip.")
    rep = diagnose(selftest="--selftest" in argv)
    print(rep.text())
    return 0 if rep.ready else 1


if __name__ == "__main__":
    sys.exit(main())
