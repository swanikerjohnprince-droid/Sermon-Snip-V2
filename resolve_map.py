"""Turn a Resolve timeline snapshot + chosen tracks into real media: analysis audio, previews, clip sources.

A *snapshot* is a plain dict (JSON) describing one timeline, produced inside Resolve by
resolve_bridge.take_snapshot() or, for a loose file, by file_snapshot() below::

    {"schema": 1, "source": "resolve"|"file", "project": str, "timeline": str, "fps": float,
     "width": int, "height": int, "duration": seconds, "warnings": [str],
     "tracks": [{"type": "audio"|"video", "index": 1, "name": "A1", "enabled": bool,
                 "items": [{"i": 0, "name": str, "file": path|None, "start": s, "end": s,
                            "src_in": s, "warnings": [str]}]}]}

``start``/``end`` are seconds on the timeline (0 = first frame); ``src_in`` is where in the source
file the item begins. Everything here works in those seconds, so it is testable without Resolve.
"""
import json
import math
import os
import re
import subprocess
import wave
import tempfile
from pathlib import Path

import numpy as np

from scene_detect import probe_video

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ANALYSIS_RATE = 16000
EPS = 1e-3


class MapError(RuntimeError):
    pass


# ---------------------------------------------------------------- snapshots

def _probe_streams(path):
    """{'audio': bool, 'video': bool, 'audio_streams': [...], 'duration','width','height','fps'} via ffprobe."""
    info = probe_video(path)
    try:
        proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=index,codec_type,codec_name,channels:stream_disposition=attached_pic:stream_tags=title,language",
                               "-of", "json", str(path)], capture_output=True, text=True, creationflags=_NO_WINDOW)
    except FileNotFoundError as exc:
        raise MapError("FFprobe/FFmpeg not found. Install FFmpeg (run INSTALL_FFMPEG.bat).") from exc
    try:
        streams = json.loads(proc.stdout or "{}").get("streams", [])
    except ValueError:
        streams = []
    audio = []
    for st in streams:
        if st.get("codec_type") == "audio":
            tags = st.get("tags") or {}
            audio.append({"n": len(audio), "codec": st.get("codec_name"), "channels": st.get("channels"),
                          "title": tags.get("title") or "", "language": tags.get("language") or ""})
    has_video = any(st.get("codec_type") == "video" and not (st.get("disposition") or {}).get("attached_pic") for st in streams)
    if not info:
        dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                             capture_output=True, text=True, creationflags=_NO_WINDOW).stdout.strip()
        info = {"duration": float(dur) if dur else 0.0}
    info.update({"audio": bool(audio), "video": has_video, "audio_streams": audio})
    return info


def file_snapshot(path):
    """Snapshot for a loose audio/video file: one audio track, plus a video track if it has video."""
    path = str(Path(path))
    if not os.path.isfile(path):
        raise MapError(f"File not found: {path}")
    info = _probe_streams(path)
    dur = float(info.get("duration") or 0)
    if dur <= 0:
        raise MapError("Could not read the duration of this file. Is it a valid audio/video file?")
    name = Path(path).name

    def item(astream=0):
        return {"i": 0, "name": name, "file": path, "start": 0.0, "end": dur, "src_in": 0.0, "astream": astream, "warnings": []}
    tracks, warnings = [], []
    streams = info.get("audio_streams") or []
    for st in streams:
        label = f"Audio {st['n'] + 1}" + (f" · {st['title']}" if st["title"] else "") + (f" · {st['channels']} ch" if st["channels"] else "")
        tracks.append({"type": "audio", "index": st["n"] + 1, "name": label, "enabled": True, "items": [item(st["n"])]})
    if not streams:
        warnings.append("This file has no audio stream, so there is nothing to analyze.")
    if info["video"]:
        tracks.append({"type": "video", "index": 1, "name": "Video", "enabled": True, "items": [dict(item(), astream=None)]})
    return {"schema": 1, "source": "file", "project": "", "timeline": name, "fps": info.get("fps") or 25.0,
            "width": info.get("width") or 0, "height": info.get("height") or 0, "duration": dur,
            "warnings": warnings, "tracks": tracks}


def measure_level(path, astream=0, start=0.0, length=15.0):
    """(mean_db, peak_db) of one audio stream over a short window, via FFmpeg's volumedetect. (None, None) if unreadable."""
    try:
        proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{max(0.0, start):.2f}", "-t", f"{length:.1f}", "-i", str(path), "-map", f"0:a:{int(astream or 0)}",
                               "-af", "volumedetect", "-vn", "-f", "null", "-"], capture_output=True, text=True, timeout=60, creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None, None
    mean = re.search(r"mean_volume:\s*(-?[\d.]+|-inf)", proc.stderr)
    peak = re.search(r"max_volume:\s*(-?[\d.]+|-inf)", proc.stderr)
    val = lambda m: None if not m else (-120.0 if m.group(1) == "-inf" else float(m.group(1)))  # noqa: E731
    return val(mean), val(peak)


SPEECH_WORDS = ("mic", "voice", "pastor", "speaker", "preach", "vocal", "lav", "lapel", "dialog", "talk", "host", "pulpit", "headset")
MUSIC_WORDS = ("music", "bgm", "band", "worship", "song", "choir", "sfx", "fx", "ambient", "room", "crowd", "click", "track")


def speech_score(track):
    """Higher = more likely the preacher: loud-enough content, plus a nudge from the track's name."""
    level = track.get("level_db")
    if level is None or track.get("silent"):
        return -999.0
    name = (track.get("name") or "").lower()
    bonus = (6 if any(w in name for w in SPEECH_WORDS) else 0) - (12 if any(w in name for w in MUSIC_WORDS) else 0)
    return level + bonus


def annotate_levels(snapshot, samples=3, length=15.0, progress=None):
    """Measure every audio track so silent ones can be flagged and the likeliest speech track picked automatically."""
    tracks = [t for t in snapshot.get("tracks", []) if t["type"] == "audio"]
    for n, trk in enumerate(tracks):
        if progress:
            progress(f"Listening to {trk.get('name') or 'track'}…", n / max(1, len(tracks)))
        usable = [it for it in trk["items"] if it.get("file")]
        if not usable:
            trk["level_db"], trk["peak_db"], trk["silent"] = None, None, True
            continue
        it = max(usable, key=lambda x: x["end"] - x["start"])
        span = max(0.1, it["end"] - it["start"])
        means, peaks = [], []
        for k in range(samples):
            frac = (k + 1) / (samples + 1)
            mean, peak = measure_level(it["file"], it.get("astream") or 0, it["src_in"] + max(0.0, span * frac - length / 2), min(length, span))
            if mean is not None:
                means.append(mean)
                peaks.append(peak)
        trk["level_db"] = round(sum(means) / len(means), 1) if means else None
        trk["peak_db"] = round(max(peaks), 1) if peaks else None
        trk["silent"] = not means or max(peaks) < -50.0 or sum(means) / len(means) < -70.0
    return snapshot


def auto_select(snapshot):
    """{'audio': [index], 'video': [index]}: the likeliest speech track, and the first video track that has footage."""
    audio = [t for t in snapshot.get("tracks", []) if t["type"] == "audio" and any(i.get("file") for i in t["items"])]
    measured = [t for t in audio if "level_db" in t]
    if measured:
        best = max(measured, key=speech_score)
        picked = [best["index"]] if speech_score(best) > -900 else []
    else:
        picked = [audio[0]["index"]] if audio else []
    video = [t["index"] for t in snapshot.get("tracks", []) if t["type"] == "video" and any(i.get("file") for i in t["items"])]
    return {"audio": picked, "video": video[:1]}


def track_summary(snapshot):
    """Compact per-track info for the UI."""
    out = []
    for trk in snapshot.get("tracks", []):
        items = trk.get("items", [])
        usable = [it for it in items if it.get("file")]
        covered = sum(max(0.0, it["end"] - it["start"]) for it in usable)
        out.append({"type": trk["type"], "index": trk["index"], "name": trk.get("name") or f"{trk['type'][0].upper()}{trk['index']}",
                    "enabled": trk.get("enabled", True), "items": len(items), "usable": len(usable),
                    "covered": round(covered, 1),
                    "files": sorted({Path(it["file"]).name for it in usable})[:3],
                    "warnings": sorted({w for it in items for w in it.get("warnings", [])})[:3],
                    "level_db": trk.get("level_db"), "peak_db": trk.get("peak_db"), "silent": trk.get("silent")})
    return out


def snapshot_signature(snapshot, audio_tracks):
    """Stable text identifying the selected audio (used for transcript cache keys)."""
    parts = []
    for trk in snapshot.get("tracks", []):
        if trk["type"] == "audio" and trk["index"] in audio_tracks:
            for it in trk["items"]:
                if it.get("file"):
                    try:
                        st = os.stat(it["file"])
                        parts.append(f"{it['file']}|{st.st_size}|{int(st.st_mtime)}|{it['start']:.3f}|{it['end']:.3f}|{it['src_in']:.3f}")
                    except OSError:
                        parts.append(f"{it['file']}|missing")
    return ";".join(parts)


# ---------------------------------------------------------------- selecting what is on screen / in the mix

def _tracks(snapshot, kind, indices):
    return [t for t in snapshot.get("tracks", []) if t["type"] == kind and t["index"] in set(indices)]


def video_pieces(snapshot, indices, t0, t1):
    """Source pieces that make up the picture between t0 and t1 on the chosen video tracks.

    Where tracks overlap the HIGHEST track wins, as in Resolve. Gaps come back as {"gap": True}.
    """
    items = [(t["index"], it) for t in _tracks(snapshot, "video", indices) for it in t["items"]
             if it.get("file") and it["end"] > t0 + EPS and it["start"] < t1 - EPS]
    cuts = {t0, t1}
    for _, it in items:
        cuts.update((min(max(it["start"], t0), t1), min(max(it["end"], t0), t1)))
    edges = sorted(cuts)
    pieces = []
    for a, b in zip(edges, edges[1:]):
        if b - a < EPS:
            continue
        cover = [(ti, it) for ti, it in items if it["start"] <= a + EPS and it["end"] >= b - EPS]
        if not cover:
            piece = {"gap": True, "tl_start": a, "tl_end": b}
        else:
            ti, it = max(cover, key=lambda x: x[0])
            piece = {"gap": False, "path": it["file"], "src_start": it["src_in"] + (a - it["start"]),
                     "src_end": it["src_in"] + (b - it["start"]), "tl_start": a, "tl_end": b, "track": ti, "item": it["i"]}
        prev = pieces[-1] if pieces else None
        if (prev and not piece["gap"] and not prev["gap"] and prev["track"] == piece["track"] and prev["item"] == piece["item"]
                and abs(prev["tl_end"] - a) < EPS and abs(prev["src_end"] - piece["src_start"]) < EPS):
            prev["tl_end"], prev["src_end"] = b, piece["src_end"]
        elif prev and piece["gap"] and prev["gap"] and abs(prev["tl_end"] - a) < EPS:
            prev["tl_end"] = b
        else:
            pieces.append(piece)
    return pieces


def audio_pieces_by_track(snapshot, indices, t0, t1):
    """{track_index: [pieces]} for every chosen audio track (they are mixed, not layered)."""
    out = {}
    for trk in _tracks(snapshot, "audio", indices):
        items = sorted((it for it in trk["items"] if it.get("file") and it["end"] > t0 + EPS and it["start"] < t1 - EPS),
                       key=lambda it: it["start"])
        pieces, cursor = [], t0
        for it in items:
            a, b = max(it["start"], t0), min(it["end"], t1)
            if a > cursor + EPS:
                pieces.append({"gap": True, "tl_start": cursor, "tl_end": a})
            a = max(a, cursor)
            if b <= a + EPS:
                continue
            pieces.append({"gap": False, "path": it["file"], "src_start": it["src_in"] + (a - it["start"]),
                           "src_end": it["src_in"] + (b - it["start"]), "tl_start": a, "tl_end": b, "track": trk["index"],
                           "item": it["i"], "astream": int(it.get("astream") or 0)})
            cursor = b
        if cursor < t1 - EPS:
            pieces.append({"gap": True, "tl_start": cursor, "tl_end": t1})
        out[trk["index"]] = pieces
    return out


def pieces_files(pieces):
    return {p["path"] for p in pieces if not p.get("gap")}


# ---------------------------------------------------------------- analysis audio

def render_analysis_audio(snapshot, audio_tracks, out_wav, progress=None, cancel=None):
    """Mix the chosen audio tracks of the whole timeline into one 16 kHz mono 16-bit WAV.

    Each item is decoded with FFmpeg straight into a disk-backed buffer, so memory stays small for
    long sermons. Returns the duration in seconds. Only the chosen tracks are heard, which is the
    whole point: pick the preacher's microphone, leave out the music.
    """
    tracks = _tracks(snapshot, "audio", audio_tracks)
    items = [(t, it) for t in tracks for it in t["items"] if it.get("file")]
    if not items:
        raise MapError("The chosen audio tracks contain no media files Sermon-Snip can read "
                       "(compound or nested clips have no file path).")
    duration = max(it["end"] for _, it in items)
    frames_out = int(math.ceil(duration * ANALYSIS_RATE))         # exactly the timeline length: no padding leaks into the duration
    total = frames_out + ANALYSIS_RATE                              # scratch slack so an item that overruns by rounding cannot index past the buffer
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".f32")
    tmp.close()
    try:
        buf = np.memmap(tmp.name, dtype=np.float32, mode="w+", shape=(total,))
        for n, (trk, it) in enumerate(items):
            if cancel is not None and cancel.is_set():
                from tools import Cancelled
                raise Cancelled()
            if progress:
                progress(f"Reading audio {n + 1}/{len(items)}: {Path(it['file']).name}", n / len(items))
            _decode_into(buf, it, total)
        buf.flush()
        np.clip(buf, -1.0, 1.0, out=buf)
        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(ANALYSIS_RATE)
            step = ANALYSIS_RATE * 60
            for lo in range(0, frames_out, step):
                wf.writeframes((buf[lo:min(lo + step, frames_out)] * 32767.0).astype("<i2").tobytes())
        del buf
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return duration


def _decode_into(buf, item, total):
    length = item["end"] - item["start"]
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, item['src_in']):.3f}", "-t", f"{length:.3f}", "-i", item["file"],
           "-map", f"0:a:{int(item.get('astream') or 0)}", "-vn", "-ac", "1", "-ar", str(ANALYSIS_RATE), "-f", "s16le", "-"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
    except FileNotFoundError as exc:
        raise MapError("FFmpeg not found. Install FFmpeg and add it to PATH.") from exc
    pos = int(round(item["start"] * ANALYSIS_RATE))
    leftover = b""
    while True:
        chunk = proc.stdout.read(1 << 20)
        if not chunk:
            break
        chunk = leftover + chunk
        usable = len(chunk) // 2 * 2
        leftover = chunk[usable:]
        samples = np.frombuffer(chunk[:usable], dtype="<i2").astype(np.float32) / 32768.0
        end = min(pos + len(samples), total)
        if end > pos:
            buf[pos:end] += samples[:end - pos]
        pos += len(samples)
    proc.stdout.close()
    err = proc.stderr.read().decode("utf-8", "replace")
    proc.stderr.close()
    if proc.wait() != 0:
        raise MapError(f"FFmpeg could not decode {Path(item['file']).name}:\n{err[-400:]}")


# ---------------------------------------------------------------- rendering ranges

def _run(cmd, cancel=None):
    from tools import run_cancelable, ToolError
    code, tail = run_cancelable(cmd, cancel=cancel)
    if code != 0:
        raise ToolError("FFmpeg failed:\n" + "\n".join(tail[-6:]))


def _has_audio(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, creationflags=_NO_WINDOW).stdout
    except FileNotFoundError:
        return False
    return bool(out.strip())


def render_video_range(pieces, out_path, size=None, fps=None, crf=20, preset="veryfast", audio=True,
                       vertical_center=False, audio_pieces=None, cancel=None):
    """Render timeline pieces to one MP4.

    pieces:        from video_pieces(); several pieces are joined, gaps become black + silence.
    size:          (w, h) to fit into (letterboxed); default = the first piece's size.
    vertical_center: crop the centre to 9:16 at 1080x1920 (fallback when verthor is unavailable).
    audio_pieces:  {track: pieces} from audio_pieces_by_track(); if given, THAT mix replaces the video's own audio.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    real = [p for p in pieces if not p.get("gap")]
    if not real:
        raise MapError("There is no video on the chosen video tracks at this moment.")
    first = probe_video(real[0]["path"])
    w, h = size or (first.get("width") or 1280, first.get("height") or 720)
    w, h = int(w) // 2 * 2, int(h) // 2 * 2
    fps = float(fps or first.get("fps") or 25.0)
    total = sum(p["tl_end"] - p["tl_start"] for p in pieces)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    chains, labels_v, labels_a, k = [], [], [], 0
    use_own_audio = audio and audio_pieces is None
    for p in pieces:
        d = p["tl_end"] - p["tl_start"]
        if p["gap"]:
            cmd += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", f"color=c=black:s={w}x{h}:r={fps}"]
            chains.append(f"[{k}:v]setsar=1,format=yuv420p[v{k}]")
            vk = k
            k += 1
            if use_own_audio:
                cmd += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
                chains.append(f"[{k}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{k}]")
                labels_a.append(f"[a{k}]")
                k += 1
            labels_v.append(f"[v{vk}]")
            continue
        cmd += ["-ss", f"{max(0.0, p['src_start']):.3f}", "-t", f"{d:.3f}", "-i", p["path"]]
        chains.append(f"[{k}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p[v{k}]")
        labels_v.append(f"[v{k}]")
        if use_own_audio:
            if _has_audio(p["path"]):
                chains.append(f"[{k}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{k}]")
                labels_a.append(f"[a{k}]")
                k += 1
            else:
                k += 1
                cmd += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
                chains.append(f"[{k}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{k}]")
                labels_a.append(f"[a{k}]")
                k += 1
        else:
            k += 1
    n = len(labels_v)
    tail_v = "[vout]"
    final_chain = ["".join(labels_v) + f"concat=n={n}:v=1:a=0[vcat]"]
    vf = "[vcat]null[vout]"
    if vertical_center:
        vf = "[vcat]crop=w='min(iw,ih*9/16)':h=ih:x='(iw-out_w)/2':y=0,scale=1080:1920:flags=lanczos,setsar=1[vout]"
    if use_own_audio:
        final_chain = ["".join(f"{v}{a}" for v, a in zip(labels_v, labels_a)) + f"concat=n={n}:v=1:a=1[vcat][acat]"]
    graph = ";".join(chains + final_chain + [vf])
    maps = ["-map", tail_v]
    if use_own_audio:
        maps += ["-map", "[acat]"]
    elif audio_pieces:
        # replace the picture's own sound with the chosen audio tracks
        a_cmd, a_graph = _audio_mix_inputs(audio_pieces, total, first_input=k)
        cmd += a_cmd
        graph += ";" + a_graph
        maps += ["-map", "[amix]"]
    cmd += ["-filter_complex", graph] + maps + ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if use_own_audio or audio_pieces:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    cmd += ["-t", f"{total:.3f}", "-movflags", "+faststart", str(out_path)]
    _run(cmd, cancel)
    return out_path


def _audio_mix_inputs(audio_pieces, total, first_input):
    """FFmpeg inputs + filter graph that builds one mixed audio stream labelled [amix]."""
    cmd, chains, track_labels, k = [], [], [], first_input
    for tidx, pieces in audio_pieces.items():
        labels = []
        for p in pieces:
            d = p["tl_end"] - p["tl_start"]
            if p["gap"]:
                cmd += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
            else:
                cmd += ["-ss", f"{max(0.0, p['src_start']):.3f}", "-t", f"{d:.3f}", "-i", p["path"]]
            sel = "" if p["gap"] else f":{int(p.get('astream') or 0)}"
            chains.append(f"[{k}:a{sel}]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[t{k}]")
            labels.append(f"[t{k}]")
            k += 1
        chains.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[trk{tidx}]")
        track_labels.append(f"[trk{tidx}]")
    if len(track_labels) == 1:
        chains.append(f"{track_labels[0]}anull[amix]")
    else:
        chains.append("".join(track_labels) + f"amix=inputs={len(track_labels)}:duration=longest:normalize=0[amix]")
    return cmd, ";".join(chains)


def render_audio_range(audio_pieces, out_path, cancel=None):
    """Render the mixed chosen audio tracks for a range to an AAC file (used for audio-only previews)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = max((p["tl_end"] for ps in audio_pieces.values() for p in ps), default=0) - min((p["tl_start"] for ps in audio_pieces.values() for p in ps), default=0)
    cmd, graph = _audio_mix_inputs(audio_pieces, total, first_input=0)
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + cmd + ["-filter_complex", graph, "-map", "[amix]",
          "-c:a", "aac", "-b:a", "128k", "-t", f"{total:.3f}", "-movflags", "+faststart", str(out_path)], cancel)
    return out_path


def replace_audio(video_path, audio_pieces, out_path, cancel=None):
    """Copy the picture of video_path and swap in the mixed chosen audio tracks."""
    total = max((p["tl_end"] for ps in audio_pieces.values() for p in ps), default=0) - min((p["tl_start"] for ps in audio_pieces.values() for p in ps), default=0)
    cmd, graph = _audio_mix_inputs(audio_pieces, total, first_input=1)
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_path)] + cmd +
         ["-filter_complex", graph, "-map", "0:v", "-map", "[amix]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
          "-t", f"{total:.3f}", "-movflags", "+faststart", str(out_path)], cancel)
    return Path(out_path)
