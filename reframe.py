"""Subject-tracking 9:16 reframing through the `verthor` CLI (KazKozDev/auto-vertical-reframe, MIT)."""
import re
from pathlib import Path

from tools import ToolError, run_cancelable

PRESETS = ["talking_head", "sports", "pets", "cars", "movie"]
_PROGRESS = re.compile(r"Processed\s+(\d+)\s*/\s*(\d+)")


def reframe_clip(verthor_cmd, src, start, end, out_path, preset="talking_head", fast=False,
                 encoder="libx264", crf=18, progress=None, cancel=None, device=None, ffmpeg_preset=None):
    """Render src[start:end] as a subject-tracked vertical MP4 (original audio kept).

    verthor extracts the time window itself (--start/--duration), so we hand it the ORIGINAL
    file rather than a pre-cut clip: one fewer lossy generation and no temp file of ours.
    `--video-encoder` defaults to Apple's VideoToolbox in verthor, so we pass libx264 explicitly.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    cmd = list(verthor_cmd) + [str(src), str(out_path), "--preset", preset,
                               "--start", f"{start:.3f}", "--duration", f"{end - start:.3f}",
                               "--video-encoder", encoder, "--crf", str(crf)]
    if fast:
        cmd.append("--fast")
    if device == "cuda":
        cmd += ["--device", "0"]            # first NVIDIA GPU (verthor's default is auto-select; this makes it explicit)
    elif device == "cpu":
        cmd += ["--device", "cpu"]
    if ffmpeg_preset:
        cmd += ["--preset-ffmpeg", ffmpeg_preset]

    def on_line(line):
        m = _PROGRESS.search(line)
        if m and progress:
            done, total = int(m.group(1)), max(1, int(m.group(2)))
            progress(min(1.0, done / total))

    code, tail = run_cancelable(cmd, cancel=cancel, on_line=on_line)
    if code != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise ToolError("verthor failed (exit %s):\n%s" % (code, "\n".join(tail[-12:])))
    return out_path
