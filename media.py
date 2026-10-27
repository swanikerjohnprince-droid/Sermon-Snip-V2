"""Shared helpers: FFmpeg audio extraction, SRT writing, timestamps, and the transcript cache."""
import hashlib
import math
import os
import shutil
import subprocess
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / "cache"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def local_ffmpeg_dirs(app_dir=None, env=None):
    """Places FFmpeg may live without being on PATH: Sermon-Snip's own copy, a winget install, common manual spots."""
    env = os.environ if env is None else env
    app = Path(app_dir) if app_dir else Path(__file__).resolve().parent
    dirs = [app / "local_tools" / "ffmpeg" / "bin"]
    local = env.get("LOCALAPPDATA")
    if local:
        pk = Path(local) / "Microsoft" / "WinGet" / "Packages"
        if pk.is_dir():
            dirs += sorted({p.parent for p in pk.glob("Gyan.FFmpeg*/**/ffmpeg.exe")}, reverse=True)
    for base in (env.get("ProgramFiles"), "C:\\"):
        if base:
            dirs += [Path(base) / "ffmpeg" / "bin"]
    return dirs


def ensure_ffmpeg_on_path(app_dir=None, env=None):
    """If ffmpeg/ffprobe are not on PATH but exist in a known place, add that folder to this process's PATH. Returns it."""
    env = os.environ if env is None else env
    if shutil.which("ffmpeg", path=env.get("PATH", "")) and shutil.which("ffprobe", path=env.get("PATH", "")):
        return None
    for d in local_ffmpeg_dirs(app_dir, env):
        if d.is_dir() and shutil.which("ffmpeg", path=str(d)) and shutil.which("ffprobe", path=str(d)):
            env["PATH"] = str(d) + os.pathsep + env.get("PATH", "")
            return d
    return None


def stamp(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def ffmpeg_available():
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def is_pcm16_wav(path):
    import wave
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getcomptype() == "NONE" and wf.getsampwidth() == 2
    except Exception:
        return False


def prepare_audio(src, workdir, progress=None):
    """Return a WAV path the analyzer can read.

    Any audio/video file is converted with FFmpeg to 16 kHz mono 16-bit PCM (what Whisper wants).
    A plain PCM16 WAV is used as-is when FFmpeg is missing. The original is never modified.
    """
    src = Path(src)
    if shutil.which("ffmpeg"):
        if progress:
            progress("Extracting audio with FFmpeg…")
        out = Path(workdir) / "sermon_snip_audio.wav"
        proc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn",
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
        if proc.returncode != 0 or not out.exists():
            tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
            raise RuntimeError(f"FFmpeg could not read this file:\n{tail}")
        return out
    if src.suffix.lower() == ".wav" and is_pcm16_wav(src):
        return src
    raise RuntimeError("FFmpeg is required for MP4/MP3/M4A/MOV files (and non-PCM WAVs). Install FFmpeg "
                       "and add it to PATH, or choose an uncompressed 16-bit PCM WAV.")


# ---------------------------------------------------------------- cache

def cache_file(src, tag, language=None, extra=""):
    p = Path(src)
    try:
        st = p.stat()
        ident = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}|{tag}|{language or 'auto'}|{extra}"
    except OSError:
        ident = f"{p}|{tag}|{language or 'auto'}|{extra}"
    return CACHE_DIR / (hashlib.sha1(ident.encode("utf-8")).hexdigest()[:20] + ".json")


# ---------------------------------------------------------------- subtitles

def srt_time(t):
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments, start=0.0, end=None, max_words=0):
    """SRT text. With start/end only that range is kept, shifted so the clip starts at 0.

    max_words > 0 splits long lines into short caption chunks; timing is spread over the segment
    in proportion to text length (the transcript has no stored word times).
    """
    end = float("inf") if end is None else end
    blocks = []
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        s0, s1 = max(seg["start"], start) - start, min(seg["end"], end) - start
        words = seg["text"].split()
        if max_words and len(words) > max_words:
            n = math.ceil(len(words) / max_words)
            per = math.ceil(len(words) / n)
            groups = [words[k:k + per] for k in range(0, len(words), per)]
        else:
            groups = [words]
        total = sum(len(" ".join(g)) for g in groups) or 1
        cursor = s0
        for g in groups:
            piece = (s1 - s0) * len(" ".join(g)) / total
            blocks.append((cursor, cursor + piece, " ".join(g)))
            cursor += piece
    return "".join(f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{t}\n\n" for n, (a, b, t) in enumerate(blocks, 1))


ensure_ffmpeg_on_path()      # make a private/winget FFmpeg visible to every subprocess call in this process
