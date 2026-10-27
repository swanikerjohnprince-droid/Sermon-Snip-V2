"""Drive the AutoSubs CLI (tmoroney/auto-subs, MIT) for transcripts and caption SRTs.

Windows note: AutoSubs release builds are GUI-subsystem executables, so when we launch them their
stdout/stderr may go nowhere. We therefore ALWAYS ask for `-o <file>` and judge success by exit code
plus the file existing — never by parsing stdout.
"""
import json
from pathlib import Path

from tools import ToolError, run_cancelable

# Short, punchy lines suit vertical video.
VERTICAL_CAPTION_FLAGS = ["--density", "custom", "--max-lines", "1", "--max-words-per-line", "4"]


def run_autosubs(autosubs_cmd, media, out_path, model="small", language=None, extra=(), cancel=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    fmt = {".srt": "srt", ".json": "json", ".vtt": "vtt", ".txt": "text"}.get(out_path.suffix.lower(), "srt")
    cmd = list(autosubs_cmd) + [str(media), "-m", model, "-f", fmt, "-o", str(out_path)]
    if language and language != "auto":
        cmd += ["-l", language]
    cmd += list(extra)
    code, tail = run_cancelable(cmd, cancel=cancel)
    if code != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        detail = "\n".join(tail[-6:]) or ("(AutoSubs printed nothing — on Windows its messages are hidden "
                                          "when launched from another app. Run the same command in a "
                                          "terminal to see the error.)")
        raise ToolError(f"AutoSubs failed (exit {code}) for {Path(media).name}:\n{detail}")
    return out_path


def caption_srt(autosubs_cmd, clip, srt_path, model="small", language=None, cancel=None):
    """Transcribe a finished clip so caption times match the clip exactly (t=0 at clip start)."""
    return run_autosubs(autosubs_cmd, clip, srt_path, model, language, VERTICAL_CAPTION_FLAGS, cancel)


def segments_from_json(data):
    """Convert an AutoSubs transcript JSON (see transcript_types.rs) to Sermon-Snip segments.

    Prefers `originalSegments` (raw engine segments, closest to Whisper's) over `segments`
    (re-wrapped for display, which would chop sentences into tiny cues).
    """
    raw = data.get("originalSegments") or data.get("segments") or []
    rows = []
    for seg in raw:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        row = {"start": float(seg["start"]), "end": float(seg["end"]), "text": text}
        if seg.get("words"):
            row["words"] = [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"])}
                            for w in seg["words"]]
        rows.append(row)
    return rows


def transcribe_autosubs(autosubs_cmd, audio_path, model="small", language=None, progress=None,
                        cancel=None, extra=(), **_ignored):
    """Whole-sermon transcription through AutoSubs -> (segments, language). The caller caches."""
    note = progress or (lambda *a: None)
    note("AutoSubs is transcribing… (first run downloads the model; long sermons take a while)", None)
    json_path = Path(audio_path).with_suffix(".autosubs.json")
    run_autosubs(autosubs_cmd, audio_path, json_path, model, language, extra, cancel=cancel)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    json_path.unlink(missing_ok=True)
    rows = segments_from_json(data)
    if not rows:
        raise ToolError("AutoSubs returned no speech for this recording.")
    return rows, data.get("language", "unknown")
