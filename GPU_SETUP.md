# NVIDIA GPU setup for Sermon-Snip

**Short version:** run `INSTALL_AND_RUN.bat`. If it sees an NVIDIA driver it installs everything the GPU
needs into the app's own environment (a one-time ~1.3 GB download). Then run `CHECK_GPU.bat` (or click
**Check GPU** in the app) to prove it works. You do **not** need to install the CUDA Toolkit or cuDNN.

Sermon-Snip still never switches to CPU silently. If the GPU fails it shows a report and offers
**Run on CPU this time**, **Install GPU libraries**, or **Use AutoSubs engine** (see below).

## What was going wrong (and what changed)

This was diagnosed from the real `ctranslate2 4.8.3` Windows wheel, not guessed.

| Cause | Detail | Fix |
|---|---|---|
| Wrong lookup mechanism | CTranslate2 loads `cublas64_12.dll` itself at run time (`LoadLibraryA`, then `%CUDA_PATH%\bin`). The old code registered folders only with Python's `os.add_dll_directory`, which that loader does not use. | `gpu_setup.py` now prepends the folders to `PATH`, points `CUDA_PATH` at them, calls `add_dll_directory`, **and preloads each DLL by absolute path** (once loaded, later bare-name loads resolve to it without searching). |
| Nothing installed the libraries | The app relied on a manual CUDA Toolkit + cuDNN install. | `requirements-gpu.txt` installs NVIDIA's own pip packages (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`). |
| The CUDA 13 trap | NVIDIA's download page now offers CUDA **13**, which contains `cublas64_13.dll`. CTranslate2 needs the **12** file. | Diagnosis names this case explicitly. The pip route avoids it. |
| cuDNN version mismatch | CTranslate2 4.8.3 bundles cuDNN **9.10.2.21**. A different cuDNN's sub-libraries next to it can fail with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`. | `nvidia-cudnn-cu12` is pinned to exactly `9.10.2.21`, and `ctranslate2` is pinned to `4.8.3`. Keep them in step if you upgrade either. |
| Incomplete search | The cuDNN 9 installer puts DLLs in `C:\Program Files\NVIDIA\CUDNN\v9.x\bin\12.x\`, and user-site packages were never scanned. | Both are searched now. |

## Check it

```
CHECK_GPU.bat
```
Runs `python gpu_setup.py --selftest`, which checks the driver, finds cuBLAS/cuDNN, loads them, asks
CTranslate2 for the device, and then runs a **real tiny model on the GPU** (first run downloads 75 MB).
It saves `gpu_report.txt`; send me that file if anything still fails.

## Reading a failure

| You see | Meaning | Do this |
|---|---|---|
| `nvidia-smi was not found` | No NVIDIA driver | Install the current driver, restart Windows |
| `cublas64_12.dll ... not found` | cuBLAS 12 missing (or only CUDA 13 installed) | `INSTALL_GPU_LIBRARIES.bat` |
| `cudnn_ops64_9.dll` / `cudnn_graph64_9.dll` | cuDNN 9 missing | `INSTALL_GPU_LIBRARIES.bat` |
| `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` | cuDNN from another release is being picked up | `INSTALL_GPU_LIBRARIES.bat`; remove old cuDNN folders from `PATH` |
| `[WinError 126]` while loading a DLL | A dependency is missing | Install the Microsoft Visual C++ 2015-2022 x64 redistributable |
| `out of memory` | Model too big for VRAM | Use `medium` or `distil-large-v3` (the RTX 4050 has 6 GB) |

Never download single DLLs from "DLL sites", and don't copy DLLs into Python folders.

## Skip CUDA entirely: the AutoSubs engine

AutoSubs transcribes through whisper.cpp built with **Vulkan** on Windows (per its `Cargo.toml`:
`windows = ["vulkan", "directml"]`), which runs on your RTX 4050 using only the normal graphics driver.
Choose **Engine: AutoSubs** in Sermon-Snip. No CUDA libraries involved.

## What is and isn't verified

- Verified: the DLL names, load behaviour and pinned versions were read from the real wheels; pip
  resolves `requirements-gpu.txt` cleanly for Windows / Python 3.12; the discovery/ordering logic is
  unit-tested against simulated Windows folder layouts (including the CUDA 13 trap and the cuDNN installer layout).
- **Not** verified: running on an actual RTX 4050. That is what `CHECK_GPU.bat` is for.

## verthor (reframing) also needs the GPU build of PyTorch

This is separate from the transcription libraries above. verthor picks CUDA automatically, but only if its PyTorch is a CUDA build, and PyPI's Windows
PyTorch is CPU-only. `INSTALL_VERTHOR.bat` installs the CUDA build when it sees an NVIDIA driver, and Step 3 of the app shows whether it worked
(**"verthor runs on the CPU"** means it didn't) with an **Enable GPU** button. Manually: `python verthor_gpu.py` (status) / `python verthor_gpu.py --install`.
The wheel folder (`cu126`, `cu128` or `cu130`) is chosen from the CUDA version your driver reports; a driver older than CUDA 12.6 must be updated first.
