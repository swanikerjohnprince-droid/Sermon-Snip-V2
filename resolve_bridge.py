# Sermon-Snip <-> DaVinci Resolve bridge.
#
# These functions take Resolve's `resolve` scripting object, so the same code runs two ways:
#   * inside Resolve, as the generated scripts SermonSnip_Export / SermonSnip_Apply / SermonSnip_Import
#     (Workspace > Scripts), which Sermon-Snip installs for you; or
#   * from the Sermon-Snip app itself when Resolve Studio allows external scripting.
#
# Keep this file plain Python 3.6-compatible: it is pasted into Resolve's embedded interpreter.
# Lessons from AutoSubs' Resolve code are applied on purpose:
#   - AppendToTimeline rejects non-integer frame values and fails the WHOLE batch -> always int().
#   - AppendToTimeline can return "dead" handles for blocked appends -> probe with GetStart().
#   - Item GetStart()/GetEnd() are absolute frames (include the timeline start, e.g. 01:00:00:00).
import json
import os
import re
import time
import traceback

HANDOFF_DIR = os.path.join(os.path.expanduser("~"), "SermonSnip")
SCHEMA = 1


# ------------------------------------------------------------------ small helpers

def notify(text):
    print("Sermon-Snip: " + text)
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Sermon-Snip", text)
        root.destroy()
    except Exception:
        pass


def read_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, data):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)      # readers never see a half-written file
    return path


def parse_fps(value, default=24.0):
    """'23.976', '29.97 DF', 25, None -> float"""
    try:
        match = re.search(r"[0-9]+(?:\.[0-9]+)?", str(value))
        number = float(match.group(0)) if match else default
        return number if number > 0 else default
    except Exception:
        return default


def call(obj, name, *args):
    """obj.name(*args) -> value, or None when missing / raising / returning nothing."""
    method = getattr(obj, name, None)
    if method is None:
        return None
    try:
        return method(*args)
    except Exception:
        return None


def is_live(item):
    """AppendToTimeline can hand back truthy dead handles; a real placed item answers GetStart()."""
    return item is not None and call(item, "GetStart") is not None


def frames(seconds, fps):
    return int(round(float(seconds) * float(fps)))


# ------------------------------------------------------------------ snapshot (read the timeline)

def take_snapshot(resolve):
    project = resolve.GetProjectManager().GetCurrentProject()
    if not project:
        raise RuntimeError("Open a project in Resolve first.")
    timeline = project.GetCurrentTimeline()
    if not timeline:
        raise RuntimeError("Open the sermon timeline in Resolve first.")
    fps = parse_fps(timeline.GetSetting("timelineFrameRate"))
    start_frame = call(timeline, "GetStartFrame") or 0
    end_frame = call(timeline, "GetEndFrame") or start_frame
    snap = {
        "schema": SCHEMA, "app": "Sermon-Snip", "source": "resolve", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "resolve_version": str(call(resolve, "GetVersionString") or ""),
        "project": str(call(project, "GetName") or ""), "timeline": str(call(timeline, "GetName") or ""),
        "fps": fps, "start_frame": int(start_frame),
        "width": int(parse_fps(timeline.GetSetting("timelineResolutionWidth"), 0)),
        "height": int(parse_fps(timeline.GetSetting("timelineResolutionHeight"), 0)),
        "duration": max(0.0, (end_frame - start_frame) / fps), "warnings": [], "tracks": [],
    }
    for kind in ("video", "audio"):
        count = int(call(timeline, "GetTrackCount", kind) or 0)
        for index in range(1, count + 1):
            enabled = call(timeline, "GetIsTrackEnabled", kind, index)
            track = {"type": kind, "index": index, "name": str(call(timeline, "GetTrackName", kind, index) or ""),
                     "enabled": True if enabled is None else bool(enabled), "items": []}
            for k, item in enumerate(call(timeline, "GetItemListInTrack", kind, index) or []):
                track["items"].append(_describe_item(item, k, fps, start_frame))
            snap["tracks"].append(track)
    unusable = sum(1 for t in snap["tracks"] for it in t["items"] if not it["file"])
    if unusable:
        snap["warnings"].append("%d timeline item(s) have no media file Sermon-Snip can read (compound, nested, "
                                "generated or multicam clips). They are skipped." % unusable)
    return snap


def _describe_item(item, k, tl_fps, tl_start_frame):
    warnings = []
    start = call(item, "GetStart")
    end = call(item, "GetEnd")
    mpi = call(item, "GetMediaPoolItem")
    path = call(mpi, "GetClipProperty", "File Path") if mpi else None
    clip_fps = parse_fps(call(mpi, "GetClipProperty", "FPS"), 0) if mpi else 0
    left = call(item, "GetLeftOffset")
    source_start = call(item, "GetSourceStartFrame")
    source_end = call(item, "GetSourceEndFrame")
    src_frames = left if left is not None else source_start
    cfps = clip_fps or tl_fps
    if not path:
        warnings.append("no file path (compound/nested/generated clip)")
    if clip_fps and abs(clip_fps - tl_fps) > 0.01:
        warnings.append("clip is %g fps on a %g fps timeline: verify the preview matches" % (clip_fps, tl_fps))
    if src_frames is None:
        warnings.append("Resolve did not report the clip's in-point; assuming it starts at the beginning of the file")
    if source_start is not None and source_end is not None and start is not None and end is not None:
        if abs((source_end - source_start) - (end - start)) > 2:
            warnings.append("looks retimed (speed change): timing may be off")
    return {
        "i": k, "name": str(call(item, "GetName") or ""), "file": str(path) if path else None,
        "start": ((start or 0) - tl_start_frame) / tl_fps, "end": ((end or 0) - tl_start_frame) / tl_fps,
        "src_in": float(src_frames or 0) / cfps, "warnings": warnings,
        "raw": {"start": start, "end": end, "left_offset": left, "source_start_frame": source_start,
                "source_end_frame": source_end, "clip_fps": clip_fps, "timeline_fps": tl_fps},
    }


# ------------------------------------------------------------------ apply (cut moments into timelines)

def _find_timeline(project, name):
    current = project.GetCurrentTimeline()
    if current and (not name or call(current, "GetName") == name):
        return current
    for n in range(1, int(call(project, "GetTimelineCount") or 0) + 1):
        candidate = call(project, "GetTimelineByIndex", n)
        if candidate and call(candidate, "GetName") == name:
            call(project, "SetCurrentTimeline", candidate)
            return candidate
    return current


def _unique_timeline(pool, name):
    for n in range(1, 60):
        candidate = name if n == 1 else "%s (%d)" % (name, n)
        timeline = pool.CreateEmptyTimeline(candidate)
        if timeline:
            return timeline
    return None


def apply_cut_job(resolve, job):
    """Create one timeline per moment containing just that stretch of the chosen tracks.

    job = {"job_id", "timeline", "bin": str, "moments": [{"name", "placements": [
              {"type": "video"|"audio", "track": 1, "item": 0, "src_start": s, "src_end": s, "rec": s}]}]}
    The clips are the ORIGINAL media-pool clips, trimmed, so nothing is rendered or re-encoded.
    """
    project = resolve.GetProjectManager().GetCurrentProject()
    pool = project.GetMediaPool()
    source = _find_timeline(project, job.get("timeline"))
    if not source:
        raise RuntimeError("Could not find the timeline '%s' in Resolve." % job.get("timeline"))
    items_cache = {}

    def source_item(kind, track, index):
        key = (kind, track)
        if key not in items_cache:
            items_cache[key] = call(source, "GetItemListInTrack", kind, track) or []
        lst = items_cache[key]
        return lst[index] if 0 <= index < len(lst) else None

    root = pool.GetRootFolder()
    folder = call(pool, "AddSubFolder", root, job.get("bin") or "Sermon-Snip moments")
    if folder:
        call(pool, "SetCurrentFolder", folder)
    report = {"job_id": job.get("job_id"), "created": [], "problems": []}
    for moment in job.get("moments", []):
        timeline = _unique_timeline(pool, moment["name"])
        if not timeline:
            report["problems"].append("could not create timeline '%s'" % moment["name"])
            continue
        for kind in ("video", "audio"):
            wanted = max([p["track"] for p in moment["placements"] if p["type"] == kind] or [0])
            while int(call(timeline, "GetTrackCount", kind) or 0) < wanted:
                if call(timeline, "AddTrack", kind) is None and int(call(timeline, "GetTrackCount", kind) or 0) < wanted:
                    report["problems"].append("could not add %s track %d" % (kind, wanted))
                    break
        call(project, "SetCurrentTimeline", timeline)      # AppendToTimeline targets the current timeline
        tl_fps = parse_fps(call(timeline, "GetSetting", "timelineFrameRate"))
        base = int(call(timeline, "GetStartFrame") or 0)
        clips, labels = [], []
        for p in moment["placements"]:
            src_item = source_item(p["type"], p["track"], p["item"])
            mpi = call(src_item, "GetMediaPoolItem") if src_item else None
            if not mpi:
                report["problems"].append("%s: no media-pool clip for %s track %s item %s" % (moment["name"], p["type"], p["track"], p["item"]))
                continue
            cfps = parse_fps(call(mpi, "GetClipProperty", "FPS"), tl_fps)
            start = frames(p["src_start"], cfps)
            end = max(start + 1, frames(p["src_end"], cfps))
            clips.append({"mediaPoolItem": mpi, "mediaType": 1 if p["type"] == "video" else 2, "startFrame": start,
                          "endFrame": end, "trackIndex": int(p["track"]), "recordFrame": base + frames(p["rec"], tl_fps)})
            labels.append("%s%d" % (p["type"][0].upper(), p["track"]))
        placed = call(pool, "AppendToTimeline", clips) if clips else []
        good = [x for x in (placed or []) if is_live(x)]
        if clips and len(good) != len(clips):
            # batch refused or partly dead: retry one by one so we place whatever Resolve will take
            good = []
            for clip in clips:
                result = call(pool, "AppendToTimeline", [clip])
                good += [x for x in (result or []) if is_live(x)]
        if clips and not good:
            report["problems"].append("Resolve placed no clips on '%s' (are the target tracks locked?)" % moment["name"])
        elif len(good) < len(clips):
            report["problems"].append("%s: placed %d of %d clips" % (moment["name"], len(good), len(clips)))
        report["created"].append({"name": moment["name"], "clips": len(good)})
    call(project, "SetCurrentTimeline", source)      # leave the user on their sermon timeline, not on the last new one
    return report


# ------------------------------------------------------------------ import finished clips

def import_package(resolve, manifest_path):
    """Import a Sermon-Snip export folder (manifest.json) as one timeline per finished clip."""
    manifest = read_json(manifest_path)
    folder_on_disk = os.path.dirname(manifest_path)
    clips = manifest.get("clips", [])
    if not clips:
        raise RuntimeError("The manifest has no clips.")
    project = resolve.GetProjectManager().GetCurrentProject()
    if not project:
        raise RuntimeError("Open a project first.")
    pool = project.GetMediaPool()
    root = pool.GetRootFolder()
    folder = call(pool, "AddSubFolder", root, "Sermon-Snip " + time.strftime("%Y-%m-%d %H%M")) or root
    call(pool, "SetCurrentFolder", folder)
    paths = [os.path.join(folder_on_disk, c["file"]) for c in clips if os.path.exists(os.path.join(folder_on_disk, c["file"]))]
    imported = pool.ImportMedia(paths) or []
    by_name = {}
    for entry in imported:
        name = call(entry, "GetClipProperty", "File Name")
        if name:
            by_name[name] = entry
    made, problems = 0, []
    for clip in clips:
        entry = by_name.get(clip["file"])
        if not entry:
            problems.append("%s was not imported" % clip["file"])
            continue
        timeline = _unique_timeline(pool, ("%02d %s" % (clip.get("index", 0), (clip.get("title") or "Highlight").strip()))[:60])
        if not timeline:
            problems.append("could not create a timeline for " + clip["file"])
            continue
        if clip.get("width") and clip.get("height"):
            # Set while the timeline is still empty; Resolve may refuse some changes once it has clips.
            call(timeline, "SetSetting", "useCustomSettings", "1")
            call(timeline, "SetSetting", "timelineResolutionWidth", str(clip["width"]))
            call(timeline, "SetSetting", "timelineResolutionHeight", str(clip["height"]))
            if clip.get("fps"):
                call(timeline, "SetSetting", "timelineFrameRate", "%g" % clip["fps"])
        call(project, "SetCurrentTimeline", timeline)
        placed = call(pool, "AppendToTimeline", [entry])
        if placed and any(is_live(x) for x in placed):
            made += 1
        else:
            problems.append("could not append " + clip["file"])
    return {"imported": made, "total": len(clips), "problems": problems, "folder": folder_on_disk}


def safe(fn, *args):
    """Run a bridge function from a Resolve script and report any failure in a dialog."""
    try:
        return fn(*args)
    except Exception:
        notify("Sermon-Snip failed:\n" + traceback.format_exc()[-1500:])
        return None
