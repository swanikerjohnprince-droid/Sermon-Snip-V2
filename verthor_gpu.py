"""Make verthor use the NVIDIA GPU.

verthor already prefers CUDA automatically, but only if PyTorch was installed as a CUDA build. On Windows, `pip install torch`
from PyPI gives a CPU-only build, which is why reframing is slow. This module (1) checks what verthor's own PyTorch sees and
(2) installs the matching CUDA build (from download.pytorch.org) into verthor's environment.

    python verthor_gpu.py            # show status
    python verthor_gpu.py --install  # install CUDA PyTorch for this PC's driver
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

INDEX = "https://download.pytorch.org/whl/"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
PROBE = ("import json,sys\n"
         "try:\n import torch\n"
         " ok=bool(torch.cuda.is_available())\n"
         " print(json.dumps({'torch':torch.__version__,'cuda':ok,'cuda_build':torch.version.cuda,"
         "'device':torch.cuda.get_device_name(0) if ok else None}))\n"
         "except Exception as e:\n print(json.dumps({'error':str(e)[:300]}))\n")


def verthor_python(cmd):
    """The python interpreter that verthor runs under, from its command prefix."""
    if not cmd:
        return None
    if len(cmd) >= 3 and cmd[1] == "-m":
        return [cmd[0]]
    exe = Path(cmd[0])
    for name in ("python.exe", "python3.exe", "python", "python3"):
        cand = exe.parent / name
        if cand.is_file():
            return [str(cand)]
    return None


def gpu_status(cmd, timeout=120):
    """{'torch','cuda','cuda_build','device'} or {'error'}. Importing torch is slow the first time, hence the timeout."""
    py = verthor_python(cmd)
    if not py:
        return {"error": "Could not find the Python that verthor runs under."}
    try:
        out = subprocess.run(py + ["-c", PROBE], capture_output=True, text=True, timeout=timeout, creationflags=_NO_WINDOW).stdout.strip().splitlines()
        return json.loads(out[-1]) if out else {"error": "No output from the PyTorch check."}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"error": f"PyTorch check failed: {exc}"}


def driver_cuda_version():
    """The newest CUDA version the installed NVIDIA driver supports (from nvidia-smi), e.g. (12, 9), or None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        guess = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(guess) if guess.is_file() else None
    if not exe:
        return None
    try:
        text = subprocess.run([exe], capture_output=True, text=True, timeout=20, creationflags=_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def tag_for_driver(version):
    """PyTorch wheel folder for a driver's CUDA capability. A wheel built for a newer CUDA than the driver supports won't run."""
    if not version:
        return None
    if version >= (13, 0):
        return "cu130"
    if version >= (12, 8):
        return "cu128"
    if version >= (12, 6):
        return "cu126"
    return None


def install_cuda_torch(cmd, tag=None, on_line=None, cancel=None):
    from tools import run_cancelable
    py = verthor_python(cmd)
    if not py:
        raise RuntimeError("Could not find verthor's Python environment.")
    ver = driver_cuda_version()
    tag = tag or tag_for_driver(ver)
    if not tag:
        raise RuntimeError("No usable NVIDIA driver found (or it is too old for CUDA 12.6). Install the current NVIDIA driver and try again."
                           if ver is None else f"Your NVIDIA driver only supports CUDA {ver[0]}.{ver[1]}. Update the driver (needs CUDA 12.6+).")
    code, tail = run_cancelable(py + ["-m", "pip", "install", "--upgrade", "--force-reinstall", "--no-deps", "torch", "torchvision",
                                      "--index-url", INDEX + tag], cancel=cancel, on_line=on_line, tail_lines=30)
    if code != 0:
        raise RuntimeError("pip could not install the CUDA build of PyTorch:\n" + "\n".join(tail[-8:]))
    status = gpu_status(cmd)
    if not status.get("cuda"):
        raise RuntimeError(f"PyTorch ({tag}) installed but still cannot see the GPU: {status.get('error') or status}. Restart Windows and try again.")
    return status


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    from tools import find_verthor
    cmd = find_verthor()
    if not cmd:
        print("verthor not found. Run INSTALL_VERTHOR.bat first.")
        return 1
    if "--install" in argv:
        print("Installing CUDA PyTorch into verthor's environment (about 2.5 GB)...")
        try:
            print("Done:", install_cuda_torch(cmd, on_line=print))
        except RuntimeError as exc:
            print("FAILED:", exc)
            return 1
        return 0
    print(json.dumps(gpu_status(cmd), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
