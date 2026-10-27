"""Cut highlight clips with FFmpeg: optional 9:16 centre crop and burned-in captions."""
import re
import subprocess
from pathlib import Path

from media import segments_to_srt, stamp

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CAPTION_STYLE = "FontName=Arial,FontSize={size},Bold=1,Outline=2,Shadow=0,Alignment=2,MarginV={margin}"


def slug(text, limit=40):
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return (s[:limit].rstrip("-")) or "clip"


def build_filter(vertical, captions_file=None, crop_x=0.5):
    """crop_x: 0 = far left, 0.5 = centre, 1 = far right (only used for vertical)."""
    chain = []
    if vertical:
        chain.append(f"crop=w='min(iw,ih*9/16)':h=ih:x='(iw-out_w)*{crop_x:.3f}':y=0")
        chain.append("scale=1080:1920:flags=lanczos")
        chain.append("setsar=1")
    if captions_file:
        style = CAPTION_STYLE.format(size=13 if vertical else 12, margin=70 if vertical else 30)
        chain.append(f"subtitles='{captions_file}':force_style='{style}'")
    return ",".join(chain)


def cut_clip(src, start, end, mp4, vertical=False, srt_name=None, crop_x=0.5, cancel=None):
    """Centre-crop / plain cut of src[start:end] with FFmpeg. mp4 must live in the cwd-relative outdir."""
    from tools import ToolError, run_cancelable
    mp4 = Path(mp4)
    vf = build_filter(vertical, srt_name, crop_x)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end - start:.3f}"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", mp4.name]
    code, tail = run_cancelable(cmd, cancel=cancel, cwd=str(mp4.parent))
    if code != 0 or not mp4.exists():
        raise ToolError("FFmpeg failed: " + " ".join(tail[-3:]))
    return mp4


def burn_captions(mp4, srt, vertical=True, cancel=None):
    """Re-encode mp4 with srt burned in (replaces the file). Both must be in the same folder."""
    from tools import ToolError, run_cancelable
    mp4, srt = Path(mp4), Path(srt)
    tmp = mp4.with_name(mp4.stem + "_cap.mp4")
    vf = build_filter(False, srt.name)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", mp4.name, "-vf", vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "copy", "-movflags", "+faststart", tmp.name]
    code, tail = run_cancelable(cmd, cancel=cancel, cwd=str(mp4.parent))
    if code != 0 or not tmp.exists():
        raise ToolError("Caption burn-in failed: " + " ".join(tail[-3:]))
    tmp.replace(mp4)
    return mp4


def export_clips(src, highlights, transcript, outdir, vertical=True, burn_captions=False,
                 crop_x=0.5, progress=None):
    """Export each highlight as MP4 plus a matching .srt. Returns (written_paths, errors)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written, errors = [], []
    for n, h in enumerate(highlights, 1):
        base = f"{n:02d}_{stamp(h['start']).replace(':', '-')}_{slug(h.get('title', ''))}"
        mp4, srt = outdir / f"{base}.mp4", outdir / f"{base}.srt"
        srt.write_text(segments_to_srt(transcript, h["start"], h["end"], max_words=6), encoding="utf-8")
        vf = build_filter(vertical, srt.name if burn_captions else None, crop_x)
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{h['start']:.3f}", "-i", str(src), "-t", f"{h['end'] - h['start']:.3f}"]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", mp4.name]
        if progress:
            progress(f"Exporting clip {n}/{len(highlights)}…", (n - 1) / len(highlights))
        try:
            # cwd=outdir keeps the subtitles filter path trivial (no Windows drive-letter escaping).
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  cwd=str(outdir), creationflags=_NO_WINDOW)
        except FileNotFoundError:
            errors.append("FFmpeg not found. Install FFmpeg and add it to PATH.")
            break
        if proc.returncode != 0 or not mp4.exists():
            errors.append(f"Clip {n}: " + " ".join(proc.stderr.strip().splitlines()[-3:]))
        else:
            written.append(mp4)
    if progress:
        progress("Clip export finished.", 1.0)
    return written, errors
