"""Highlights -> finished vertical clips + captions + a manifest the Resolve script imports."""
import json
import time
from dataclasses import dataclass
from pathlib import Path

from media import segments_to_srt, stamp
from autosubs_cli import caption_srt
from clipper import burn_captions, cut_clip, slug
from reframe import reframe_clip
from scene_detect import probe_video
from tools import Cancelled, HANDOFF_DIR, ToolError

VERSION = "0.5"


@dataclass
class ShortsOptions:
    vertical: bool = True
    smart_reframe: bool = True          # verthor subject tracking (falls back to centre crop)
    preset: str = "talking_head"
    fast: bool = False                  # verthor --fast: much quicker on CPU, less precise
    captions: str = "autosubs"          # "autosubs" | "builtin" | "none"
    burn_captions: bool = False         # bake captions into the video (skip if styling in Resolve)
    autosubs_model: str = "small"
    language: str = None
    verthor_cmd: list = None
    autosubs_cmd: list = None
    encoder: str = "libx264"


def build_shorts(src, highlights, transcript, outdir, opts, progress=None, cancel=None):
    """Produce one folder of clips. Returns the manifest dict (also written as manifest.json).

    Each clip degrades gracefully: verthor failure -> centre crop; AutoSubs failure -> built-in SRT.
    Every fallback is recorded in the clip's `warnings` so nothing fails silently.
    """
    note = progress or (lambda *a: None)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"app": "Sermon-Snip", "version": VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source_media": str(src), "output_dir": str(outdir.resolve()), "clips": []}
    total = len(highlights)
    for n, h in enumerate(highlights, 1):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        base = f"{n:02d}_{stamp(h['start']).replace(':', '-')}_{slug(h.get('title', ''))}"
        mp4, srt = outdir / f"{base}.mp4", outdir / f"{base}.srt"
        warnings, method = [], "none"

        def sub(label, frac):
            note(f"Clip {n}/{total}: {label}", (n - 1 + (frac if frac is not None else 0.0)) / total)

        # ---- 1. video
        done = False
        if opts.vertical and opts.smart_reframe and opts.verthor_cmd:
            try:
                sub("tracking subject with verthor…", 0.0)
                reframe_clip(opts.verthor_cmd, src, h["start"], h["end"], mp4, opts.preset, opts.fast,
                             opts.encoder, progress=lambda f: sub("tracking subject with verthor…", f * 0.8),
                             cancel=cancel)
                method, done = "verthor", True
            except ToolError as exc:
                warnings.append(f"verthor failed, used centre crop instead: {exc}")
        elif opts.vertical and opts.smart_reframe and not opts.verthor_cmd:
            warnings.append("verthor not found, used centre crop")
        if not done:
            sub("cutting clip…", 0.0)
            cut_clip(src, h["start"], h["end"], mp4, vertical=opts.vertical, cancel=cancel)
            method = "center-crop" if opts.vertical else "plain-cut"

        # ---- 2. captions (timed against the finished clip, t=0 = clip start)
        engine = "none"
        if opts.captions == "autosubs" and opts.autosubs_cmd:
            try:
                sub("captioning with AutoSubs…", 0.85)
                caption_srt(opts.autosubs_cmd, mp4, srt, opts.autosubs_model, opts.language, cancel)
                engine = "autosubs"
            except ToolError as exc:
                warnings.append(f"AutoSubs failed, used built-in captions instead: {exc}")
        elif opts.captions == "autosubs":
            warnings.append("AutoSubs not found, used built-in captions")
        if opts.captions != "none" and engine == "none":
            srt.write_text(segments_to_srt(transcript, h["start"], h["end"], max_words=6), encoding="utf-8")
            engine = "builtin"
        if opts.burn_captions and engine != "none":
            try:
                sub("burning captions…", 0.95)
                burn_captions(mp4, srt, vertical=opts.vertical, cancel=cancel)
            except ToolError as exc:
                warnings.append(str(exc))

        info = probe_video(mp4)
        manifest["clips"].append({
            "index": n, "file": mp4.name, "captions": srt.name if engine != "none" else None,
            "caption_engine": engine, "reframe": method, "vertical": opts.vertical,
            "width": info.get("width"), "height": info.get("height"), "fps": info.get("fps"),
            "duration": info.get("duration"), "source_start": h["start"], "source_end": h["end"],
            "title": h.get("title", ""), "score": h.get("score"), "text": h.get("text", ""),
            "warnings": warnings})
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    note("Package ready.", 1.0)
    write_pointer("last_manifest.txt", str((outdir / "manifest.json").resolve()))
    return manifest


def write_pointer(name, text):
    """Leave a breadcrumb in ~/SermonSnip so Resolve scripts can find the latest export without a file dialog."""
    try:
        HANDOFF_DIR.mkdir(exist_ok=True)
        (HANDOFF_DIR / name).write_text(text, encoding="utf-8")
    except OSError:
        pass
