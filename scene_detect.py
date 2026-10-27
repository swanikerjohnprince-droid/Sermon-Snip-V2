import json
import re
import subprocess

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def probe_duration(path):
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True, creationflags=_NO_WINDOW)
        return float(proc.stdout.strip())
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe not found. Install FFmpeg and add its bin folder to PATH.") from exc
    except Exception as exc:
        raise RuntimeError("Could not read video duration; check FFmpeg and the file.") from exc


def has_video(path):
    """True if the file contains a real video stream (cover art doesn't count)."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
             "stream=index:stream_disposition=attached_pic", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, creationflags=_NO_WINDOW)
    except FileNotFoundError:
        return False
    rows = [r.strip() for r in proc.stdout.splitlines() if r.strip()]
    return any(not r.endswith(",1") for r in rows)


def probe_video(path):
    """Return {'width','height','fps','duration'} for the first video stream, or {} if unreadable."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, check=True, creationflags=_NO_WINDOW)
        data = json.loads(proc.stdout)
        stream = (data.get("streams") or [{}])[0]
        num, _, den = str(stream.get("avg_frame_rate", "0/1")).partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        return {"width": int(stream.get("width", 0)), "height": int(stream.get("height", 0)),
                "fps": round(fps, 3), "duration": float(data.get("format", {}).get("duration", 0) or 0)}
    except Exception:
        return {}


def detect_cuts(path, threshold=0.32):
    # Downscale first: scene scores barely change but decoding/filtering is far faster on long 1080p/4K files.
    vf = f"scale=320:-2,select='gt(scene,{threshold})',showinfo"
    cmd = ["ffmpeg", "-hide_banner", "-i", str(path), "-vf", vf, "-an", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", check=False, creationflags=_NO_WINDOW)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Install FFmpeg and add its bin folder to PATH.") from exc
    times = []
    for line in proc.stderr.splitlines():
        if "showinfo" in line and "pts_time:" in line:
            match = re.search(r"pts_time:\s*([0-9.]+)", line)
            if match:
                time = float(match.group(1))
                if not times or time - times[-1] > .15:
                    times.append(round(time, 3))
    if proc.returncode and not times:
        raise RuntimeError("FFmpeg scene detection failed:\n" + "\n".join(proc.stderr.splitlines()[-10:]))
    return times


def make_scenes(cuts, duration):
    points = sorted(set([0.0] + [t for t in cuts if .1 < t < duration - .1] + [float(duration)]))
    return [{"scene": i + 1, "start": round(points[i], 3), "end": round(points[i + 1], 3),
             "duration": round(points[i + 1] - points[i], 3)} for i in range(len(points) - 1)]
