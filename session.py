"""The Sermon-Snip workflow as one controller, independent of any UI.

    1 SOURCE   -> detect tracks that have sound, choose which to use (Resolve timeline or a file)
    2 MOMENTS  -> find the best moments (optionally re-judged by Gemini), preview, adjust, optionally cut into Resolve
    3 REFRAME  -> 9:16 subject-tracked reframing (or keep the original framing), before/after previews
    4 CAPTIONS -> AutoSubs transcription, AutoSubs presets + formatting, live caption preview
    5 EXPORT   -> render every finished clip (or press "Export all" at any point to run every remaining stage)

All long work runs in background threads and reports through ``state()`` (a JSON-able dict with a revision counter).
"""
import json
import re
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import caption_style as cs
import captions as cap
import gemini_rank
import gpu_setup
import resolve_map as rm
import verthor_gpu
from analyzer import GpuError, analyze_wav_energy, candidates, transcribe
from autosubs_cli import run_autosubs, transcribe_autosubs
from clipper import slug
from media import cache_file, ffmpeg_available, stamp
from reframe import PRESETS, reframe_clip
from scene_detect import probe_video
from tools import Cancelled, HANDOFF_DIR, ToolError, find_autosubs, find_verthor, load_config, run_cancelable, save_config

VERSION = "0.7"
DEFAULT_SETTINGS = {"engine": "whisper", "device": "cuda", "model": "large-v3", "language": "en", "vocabulary": "", "target": 60,
                    "gemini": False, "gemini_model": gemini_rank.DEFAULT_MODEL, "gemini_weight": 0.7}
DEFAULT_REFRAME = {"smart": True, "preset": "talking_head", "fast": False, "audio": "auto", "aspect": "vertical"}
DEFAULT_CAPTIONS = {"preset": "ss:bold_pop", "density": "standard", "max_words": 3, "max_chars": 0, "lines": 1, "text_case": "none",
                    "remove_punctuation": False, "size": 1.0, "highlight": "", "burn": True, "source": "autosubs", "autosubs_model": "small"}
DEFAULT_EXPORT = {"quality": "high", "nvenc": "auto", "sidecar_srt": True, "sidecar_ass": False}


class WorkflowError(RuntimeError):
    """A user-facing problem (shown in the UI as is)."""


def _title(text, signals=(), words=9):
    """A headline for a moment: the first sentence that contains a strong phrase, else the first sentence."""
    sentences = [x for x in re.split(r"(?<=[.?!])\s+", text.strip()) if x.strip()] or [text.strip()]
    low = [str(x).lower() for x in signals or ()]
    pick = next((x for x in sentences if any(sig in x.lower() for sig in low)), sentences[0])
    toks = pick.split()
    return " ".join(toks[:words]) + ("…" if len(toks) > words else "")


class Session:
    def __init__(self, workdir=None, link=None, tools=None, export_root=None, config=None, autosubs_presets_path=None):
        self.work = Path(workdir) if workdir else Path(HANDOFF_DIR) / "work" / time.strftime("%Y%m%d-%H%M%S")
        for sub in ("previews", "reframed", "captions", "src", "burn", "thumbs"):
            (self.work / sub).mkdir(parents=True, exist_ok=True)
        self.export_root = Path(export_root) if export_root else Path(HANDOFF_DIR) / "exports"
        self.config = config if config is not None else load_config()
        self.link = link
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.previews = ThreadPoolExecutor(max_workers=2, thread_name_prefix="preview")
        self.rev = 0
        self.snapshot = None
        self.sel = {"audio": [], "video": []}
        self.settings = dict(DEFAULT_SETTINGS, **{k: v for k, v in self.config.get("web_settings", {}).items() if k in DEFAULT_SETTINGS})
        self.reframe_opts = dict(DEFAULT_REFRAME)
        self.caption_opts = dict(DEFAULT_CAPTIONS, **{k: v for k, v in self.config.get("caption_opts", {}).items() if k in DEFAULT_CAPTIONS})
        self.export_opts = dict(DEFAULT_EXPORT, **{k: v for k, v in self.config.get("export_opts", {}).items() if k in DEFAULT_EXPORT})
        self.export_dir = None
        self.custom_export_dir = None
        self.moments, self.transcript, self.duration, self.language, self.generation = [], [], 0.0, "unknown", 0
        self.job = {"kind": None, "status": "idle", "message": "", "fraction": None, "error": None}
        self.gpu = {"status": "idle", "ready": None, "text": "", "error": None, "recover": None}
        self.vgpu = {"status": "unknown"}
        self.cut = {"status": "idle", "message": "", "result": None}
        self.imported = None
        self.last_manifest = None
        self.captions_test = None
        self.gemini_note = ""
        self._snap_mtime = None
        self._sel_touched = False
        self._nvenc = None
        self._tools_override = tools
        self._presets_path = autosubs_presets_path or self.config.get("autosubs_presets_path")
        self.presets, self.presets_info = cs.load_presets(self._presets_path)
        self.refresh_tools()

    # ------------------------------------------------------------ plumbing
    def bump(self):
        with self.lock:
            self.rev += 1

    def refresh_tools(self):
        if self._tools_override is not None:
            self.tools = dict(self._tools_override)
        else:
            self.tools = {"ffmpeg": ffmpeg_available(), "verthor": find_verthor(self.config), "autosubs": find_autosubs(self.config)}
        self.presets, self.presets_info = cs.load_presets(self._presets_path)
        self.bump()
        return self.tools

    def locate_tool(self, key, path):
        if key not in ("verthor_path", "autosubs_path", "autosubs_presets_path"):
            raise WorkflowError("Unknown tool.")
        self.config[key] = path
        if key == "autosubs_presets_path":
            self._presets_path = path
        save_config(self.config)
        self.refresh_tools()

    def _save_opts(self):
        self.config["web_settings"] = {k: self.settings[k] for k in DEFAULT_SETTINGS}
        self.config["caption_opts"], self.config["export_opts"] = dict(self.caption_opts), dict(self.export_opts)
        try:
            save_config(self.config)
        except OSError:
            pass

    def _begin(self, kind, message):
        with self.lock:
            if self.job["status"] == "running":
                raise WorkflowError("Another task is still running. Wait for it or press Cancel.")
            self.cancel_event.clear()
            self.job = {"kind": kind, "status": "running", "message": message, "fraction": None, "error": None}
        self.bump()

    def _progress(self, message=None, fraction=None):
        with self.lock:
            if message is not None:
                self.job["message"] = message
            self.job["fraction"] = fraction
        self.bump()

    def _end(self, status="done", error=None, message=None):
        with self.lock:
            self.job["status"] = status
            self.job["error"] = error
            if message is not None:
                self.job["message"] = message
            self.job["fraction"] = 1.0 if status == "done" else self.job["fraction"]
        self.bump()

    def _run(self, kind, message, fn, *args):
        """Start fn(*args) in a background thread as the one active job."""
        self._begin(kind, message)

        def runner():
            try:
                note = fn(*args)
                self._end("done", message=note if isinstance(note, str) else None)
            except Cancelled:
                self._end("cancelled", message="Cancelled.")
            except GpuError as exc:
                with self.lock:
                    self.gpu.update({"recover": {"message": str(exc), "diagnosis": exc.diagnosis}})
                self._end("error", error="NVIDIA GPU transcription could not start. See the GPU report.")
            except (WorkflowError, ToolError, rm.MapError, gemini_rank.GeminiError) as exc:
                self._end("error", error=str(exc))
            except Exception as exc:                                    # never let a worker die silently
                self._end("error", error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-900:]}")

        threading.Thread(target=runner, daemon=True, name=f"job-{kind}").start()

    def cancel(self):
        self.cancel_event.set()
        with self.lock:
            if self.job["status"] == "running":
                self.job["message"] = "Cancelling…"
        self.bump()

    def _check(self):
        if self.cancel_event.is_set():
            raise Cancelled()

    # ------------------------------------------------------------ 1. source
    def load_file(self, path):
        self._snap_mtime = None
        self.set_snapshot(rm.file_snapshot(path))

    def refresh_from_resolve(self):
        """Pick up the timeline Resolve exported. Cheap to call repeatedly: it only reloads when the file changed."""
        if not self.link:
            return False
        if not self.link.direct:
            mtime = self.link.snapshot_mtime()
            if mtime is None:
                return False
            if mtime == self._snap_mtime and self.snapshot is not None:
                return True
            snap = self.link.read_snapshot()
            if not snap:
                return False
            self._snap_mtime = mtime
        else:
            snap = self.link.read_snapshot()
        self.set_snapshot(snap)
        return True

    def set_snapshot(self, snap):
        """Adopt a snapshot, then listen to each audio track in the background so silent tracks are flagged and the likeliest
        speech track is chosen for you."""
        with self.lock:
            same = self.snapshot and (self.snapshot.get("timeline"), self.snapshot.get("source")) == (snap.get("timeline"), snap.get("source"))
            old_sel = dict(self.sel)
            self.snapshot = snap
            usable = lambda kind: [t["index"] for t in snap["tracks"] if t["type"] == kind and any(i.get("file") for i in t["items"])]  # noqa: E731
            keep = {k: [i for i in old_sel[k] if i in usable(k)] for k in ("audio", "video")} if same else {"audio": [], "video": []}
            self.sel = {"audio": keep["audio"] or usable("audio")[:1], "video": keep["video"] or usable("video")[:1]}
            if not same:
                self.moments, self.transcript = [], []
                self._sel_touched = False
            needs_levels = any(t["type"] == "audio" and "level_db" not in t for t in snap["tracks"])
            self.listening = needs_levels
        self.bump()
        if needs_levels:
            self.previews.submit(self._listen, snap, not same or not any(old_sel.values()))

    def _listen(self, snap, choose):
        try:
            rm.annotate_levels(snap)
            with self.lock:
                if snap is self.snapshot and choose and not self._sel_touched:      # never override a choice the user already made
                    self.sel = rm.auto_select(snap)
        except Exception:
            pass
        finally:
            with self.lock:
                self.listening = False
            self.bump()

    def select_tracks(self, audio, video):
        if not self.snapshot:
            raise WorkflowError("Load a timeline or file first.")
        have = {(t["type"], t["index"]) for t in self.snapshot["tracks"]}
        with self.lock:
            self.sel = {"audio": [i for i in audio if ("audio", i) in have], "video": [i for i in video if ("video", i) in have]}
            self._sel_touched = True
            self.moments = []                      # moments depend on which audio was analysed
        self.bump()

    # ------------------------------------------------------------ 2. moments
    def analyze(self, settings):
        if not self.snapshot:
            raise WorkflowError("Choose a Resolve timeline or a file first (step 1).")
        if not self.sel["audio"]:
            raise WorkflowError("Choose at least one audio track to analyze (step 1).")
        s = dict(self.settings, **{k: v for k, v in (settings or {}).items() if k in DEFAULT_SETTINGS})
        s["target"] = max(15, min(180, int(s["target"])))
        s["gemini"] = bool(s["gemini"])
        if s["engine"] == "autosubs" and not self.tools.get("autosubs"):
            raise WorkflowError("AutoSubs was not found. Install it, or use 'Locate tools', or pick the Whisper engine.")
        if s["gemini"] and not gemini_rank.get_key(self.config):
            raise WorkflowError("Gemini ranking is on but no API key is set. Add one under 'Smarter ranking', or switch it off.")
        with self.lock:
            self.settings = s
            self.gpu["recover"] = None
        self._save_opts()
        self._run("analyze", "Preparing audio…", self._analyze_worker, s)

    def _analyze_worker(self, s):
        snap, tracks = self.snapshot, list(self.sel["audio"])
        wav = self.work / "analysis.wav"
        rm.render_analysis_audio(snap, tracks, wav, progress=lambda m, f: self._progress(m, f), cancel=self.cancel_event)
        self._check()
        duration, energy = analyze_wav_energy(wav, progress=lambda f: self._progress(f"Measuring delivery energy… {f:.0%}", f))
        self._check()
        transcript, language = self._transcript(wav, s, tracks)
        self._check()
        self._progress("Ranking moments…", None)
        found = candidates(transcript, energy, duration, target_seconds=s["target"])
        found.sort(key=lambda m: m.get("score", 0), reverse=True)
        with self.lock:
            self.generation += 1
            self.transcript, self.language, self.duration = transcript, language, duration
            self.moments = self._make_moments(found)
        note = f"Found {len(found)} moments."
        if s["gemini"]:
            self._progress("Asking Gemini to judge the best moments…", None)
            note += " " + self._gemini_pass(s)
        return note

    def _make_moments(self, found):
        return [dict(m, id=i + 1, title=_title(m["text"], m.get("signals")), selected=False, ver=0, orig=(m["start"], m["end"]),
                     preview={"status": "idle"}, reframe={"status": "idle"}, captions={"status": "idle"}, cpreview={"status": "idle"},
                     final={"status": "idle"}) for i, m in enumerate(found)]

    def _gemini_pass(self, s):
        key = gemini_rank.get_key(self.config)
        with self.lock:
            pool = [dict(id=m["id"], start=m["start"], end=m["end"], text=m["text"]) for m in self.moments[:30]]
        try:
            rated = gemini_rank.rerank(pool, key, s["gemini_model"], endpoint=gemini_rank.ENDPOINT)
        except gemini_rank.GeminiError as exc:
            self.gemini_note = f"Gemini ranking skipped: {exc}"
            return self.gemini_note
        w = float(s["gemini_weight"])
        with self.lock:
            for m in self.moments:
                r = rated.get(m["id"])
                if r:
                    m["local_score"], m["ai_score"], m["ai_reason"] = m["score"], r["score"], r["reason"]
                    m["score"] = gemini_rank.blend(m["score"], r["score"], w)
                    if r["title"]:
                        m["title"] = r["title"]
            self.moments.sort(key=lambda m: m["score"], reverse=True)
            for i, m in enumerate(self.moments):
                m["id"] = i + 1
        self.gemini_note = f"Gemini judged {len(rated)} moments."
        return self.gemini_note

    def _transcript(self, wav, s, tracks):
        snap = self.snapshot
        first = next((it["file"] for t in snap["tracks"] if t["type"] == "audio" and t["index"] in tracks for it in t["items"] if it.get("file")), str(wav))
        extra = rm.snapshot_signature(snap, tracks) + "|" + s["vocabulary"]
        language = None if s["language"] in ("", "auto") else s["language"]
        cache = cache_file(first, f"web-{s['engine']}-{s['model']}", language, extra=extra)
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            self._progress("Using the saved transcript for this audio and settings.", 1.0)
            return data["segments"], data.get("language", "unknown")
        except (OSError, ValueError, KeyError):
            pass
        if s["engine"] == "autosubs":
            more = (["--no-gpu"] if s["device"] == "cpu" else []) + (["--prompt", s["vocabulary"]] if s["vocabulary"] else [])
            rows, lang = transcribe_autosubs(self.tools["autosubs"], wav, s["model"], language, lambda m, f=None: self._progress(m, f),
                                             cancel=self.cancel_event, extra=more)
        else:
            rows, lang = transcribe(wav, s["model"], progress=lambda m: self._progress(m, None), language=language,
                                    device="cpu" if s["device"] == "cpu" else "cuda", chunk_seconds=30, overlap_seconds=2,
                                    initial_prompt=s["vocabulary"] or None)
        try:
            cache.parent.mkdir(exist_ok=True)
            cache.write_text(json.dumps({"language": lang, "segments": rows}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return rows, lang

    def rerank(self, target):
        if not self.transcript:
            raise WorkflowError("Find moments first.")
        target = max(15, min(180, int(target)))
        self.settings["target"] = target
        self._run("rerank", "Re-ranking…", self._rerank_worker, target)

    def _rerank_worker(self, target):
        wav = self.work / "analysis.wav"
        _, energy = analyze_wav_energy(wav)
        found = candidates(self.transcript, energy, self.duration, target_seconds=target)
        found.sort(key=lambda m: m.get("score", 0), reverse=True)
        with self.lock:
            self.generation += 1
            self.moments = self._make_moments(found)
        return f"Found {len(found)} moments."

    def set_gemini_key(self, key):
        key = (key or "").strip()
        if key:
            self.config["gemini_key"] = key
        else:
            self.config.pop("gemini_key", None)
        save_config(self.config)
        self.bump()

    def test_gemini(self, key=None):
        names = gemini_rank.test_key((key or gemini_rank.get_key(self.config)), endpoint=gemini_rank.ENDPOINT)
        return names

    def get_moment(self, mid):
        with self.lock:
            for m in self.moments:
                if m["id"] == int(mid):
                    return m
        raise WorkflowError(f"Moment {mid} does not exist (the list changed). Refresh the page.")

    def set_selected(self, ids):
        ids = {int(i) for i in ids}
        with self.lock:
            for m in self.moments:
                m["selected"] = m["id"] in ids
        self.bump()

    def chosen(self, ids=None):
        with self.lock:
            pick = [m for m in self.moments if (m["id"] in {int(i) for i in ids} if ids is not None else m["selected"])]
        if not pick:
            raise WorkflowError("Select at least one moment first.")
        return sorted(pick, key=lambda m: m["start"])

    # ---- adjusting a moment's start/end (the cut is rebuilt from the original media, so nothing is lost)
    def adjust(self, mid, start=None, end=None, reset=False):
        m = self.get_moment(mid)
        with self.lock:
            if reset:
                start, end = m["orig"]
            start = float(m["start"] if start is None else start)
            end = float(m["end"] if end is None else end)
            limit = min([v for v in (self.duration, (self.snapshot or {}).get("duration")) if v] or [end])
            start, end = max(0.0, start), min(float(limit), end)
            if end - start < 3.0:
                raise WorkflowError("A clip must be at least 3 seconds long.")
            if end - start > 180.0:
                raise WorkflowError("A clip can be at most 3 minutes long.")
            m["start"], m["end"] = round(start, 2), round(end, 2)
            m["ver"] += 1
            m["text"] = " ".join(seg["text"] for seg in self.transcript if seg["end"] > start and seg["start"] < end) or m["text"]
            m["title"] = m["title"] or _title(m["text"])
            for key in ("preview", "reframe", "captions", "cpreview", "final"):
                m[key] = {"status": "idle"}
        self.bump()

    # ---- previews (original framing, small, with sound)
    def request_preview(self, mid):
        m = self.get_moment(mid)
        with self.lock:
            if m["preview"]["status"] in ("queued", "running", "ready"):
                return
            m["preview"] = {"status": "queued"}
            gen, ver = self.generation, m["ver"]
        self.bump()
        self.previews.submit(self._preview_worker, m["id"], gen, ver)

    def _preview_worker(self, mid, gen, ver):
        try:
            m = self.get_moment(mid)
        except WorkflowError:
            return
        if gen != self.generation or ver != m["ver"]:
            return
        with self.lock:
            m["preview"] = {"status": "running"}
        self.bump()
        try:
            out, kind = self._make_preview(m, ver)
            state = {"status": "ready", "kind": kind, "file": f"previews/{out.name}"}
        except Exception as exc:
            state = {"status": "error", "error": str(exc)[:300]}
        with self.lock:
            if gen == self.generation and ver == m["ver"]:
                m["preview"] = state
        self.bump()

    def _make_preview(self, m, ver):
        snap = self.snapshot
        name = f"o{self.generation}_{m['id']}_{ver}"
        pieces = rm.video_pieces(snap, self.sel["video"], m["start"], m["end"]) if self.sel["video"] else []
        if any(not p["gap"] for p in pieces):
            first = next(p for p in pieces if not p["gap"])
            info = probe_video(first["path"])
            h = 360
            w = int(round((info.get("width") or 640) * h / (info.get("height") or 360))) // 2 * 2
            ap = rm.audio_pieces_by_track(snap, self.sel["audio"], m["start"], m["end"])
            out = self.work / "previews" / f"{name}.mp4"
            rm.render_video_range(pieces, out, size=(w, h), crf=30, preset="ultrafast", audio_pieces=ap if self._audio_differs(pieces, ap) else None)
            return out, "video"
        ap = rm.audio_pieces_by_track(snap, self.sel["audio"], m["start"], m["end"])
        out = self.work / "previews" / f"{name}.m4a"
        rm.render_audio_range(ap, out)
        return out, "audio"

    @staticmethod
    def _audio_differs(pieces, audio_pieces):
        """True when the chosen audio comes from other files, or another audio stream, than the picture's own sound."""
        flat = [p for ps in (audio_pieces or {}).values() for p in ps if not p.get("gap")]
        if not flat:
            return False
        return any(p.get("astream") for p in flat) or not rm.pieces_files(flat) <= rm.pieces_files(pieces)

    # ---- cut into Resolve timelines
    def cut_in_resolve(self, ids=None):
        if not self.snapshot or self.snapshot.get("source") != "resolve":
            raise WorkflowError("Cutting into Resolve timelines needs a timeline loaded from Resolve (step 1).")
        picked = self.chosen(ids)
        snap, job_moments = self.snapshot, []
        for m in picked:
            placements = []
            for trk in snap["tracks"]:
                if trk["index"] not in self.sel[trk["type"]]:
                    continue
                for it in trk["items"]:
                    if it.get("file") and it["end"] > m["start"] + 1e-3 and it["start"] < m["end"] - 1e-3:
                        a, b = max(it["start"], m["start"]), min(it["end"], m["end"])
                        placements.append({"type": trk["type"], "track": trk["index"], "item": it["i"],
                                           "src_start": it["src_in"] + (a - it["start"]), "src_end": it["src_in"] + (b - it["start"]), "rec": a - m["start"]})
            job_moments.append({"name": f"{m['id']:02d} {m['title']}"[:60], "placements": placements})
        job = {"timeline": snap["timeline"], "bin": "Sermon-Snip moments " + time.strftime("%H%M"), "moments": job_moments}
        job_id, report = self.link.submit_cut(job)
        with self.lock:
            if report is not None:
                self.cut = {"status": "done", "message": f"Created {len(report['created'])} timeline(s) in Resolve.", "result": report}
            else:
                self.cut = {"status": "waiting", "job_id": job_id, "result": None,
                            "message": "Waiting for Resolve: in Resolve choose Workspace ▸ Scripts ▸ SermonSnip."}
                threading.Thread(target=self._wait_for_cut, args=(job_id,), daemon=True).start()
        self.bump()

    def _wait_for_cut(self, job_id, timeout=1800):
        end = time.time() + timeout
        while time.time() < end:
            result = self.link.read_cut_result(job_id)
            if result is not None:
                with self.lock:
                    note = f"Created {len(result['created'])} timeline(s) in Resolve."
                    if result["problems"]:
                        note += f" {len(result['problems'])} problem(s): " + "; ".join(result["problems"][:2])
                    self.cut = {"status": "done", "message": note, "result": result}
                self.bump()
                return
            time.sleep(1.0)
        with self.lock:
            self.cut = {"status": "idle", "message": "Timed out waiting for Resolve.", "result": None}
        self.bump()

    # ------------------------------------------------------------ 3. reframe
    def check_verthor_gpu(self, force=False):
        """What does verthor's own PyTorch see? Runs in the background (importing torch takes a few seconds)."""
        cmd = self.tools.get("verthor")
        if not cmd:
            self.vgpu = {"status": "missing"}
            return
        if self.vgpu.get("status") in ("checking",) or (self.vgpu.get("status") in ("gpu", "cpu") and not force):
            return
        self.vgpu = {"status": "checking"}
        self.bump()

        def work():
            st = verthor_gpu.gpu_status(cmd)
            with self.lock:
                if st.get("error"):
                    self.vgpu = {"status": "error", "error": st["error"]}
                elif st.get("cuda"):
                    self.vgpu = {"status": "gpu", "device": st.get("device"), "torch": st.get("torch")}
                else:
                    self.vgpu = {"status": "cpu", "torch": st.get("torch"), "driver_cuda": verthor_gpu.driver_cuda_version(),
                                 "can_install": bool(verthor_gpu.tag_for_driver(verthor_gpu.driver_cuda_version()))}
            self.bump()
        self.previews.submit(work)

    def enable_verthor_gpu(self):
        cmd = self.tools.get("verthor")
        if not cmd:
            raise WorkflowError("verthor is not installed.")
        self._run("verthor-gpu", "Installing CUDA PyTorch into verthor (about 2.5 GB)…", self._enable_verthor_gpu_worker, cmd)

    def _enable_verthor_gpu_worker(self, cmd):
        verthor_gpu.install_cuda_torch(cmd, on_line=lambda line: self._progress(line[:140], None), cancel=self.cancel_event)
        self.vgpu = {"status": "unknown"}
        self.check_verthor_gpu(force=True)
        return "verthor can now use your NVIDIA GPU."

    def reframe(self, ids, opts):
        if not self.sel["video"]:
            raise WorkflowError("Reframing needs a video track (step 1).")
        picked = self.chosen(ids)
        o = self._reframe_options(opts)
        with self.lock:
            for m in picked:
                m["reframe"] = {"status": "queued"}
                m["captions"], m["final"] = {"status": "idle"}, {"status": "idle"}
        self._run("reframe", f"Reframing {len(picked)} clip(s)…", self._reframe_worker, [m["id"] for m in picked], o)

    def _reframe_options(self, opts):
        o = dict(self.reframe_opts, **{k: v for k, v in (opts or {}).items() if k in DEFAULT_REFRAME})
        if o["preset"] not in PRESETS:
            o["preset"] = "talking_head"
        if o["aspect"] not in ("vertical", "original"):
            o["aspect"] = "vertical"
        self.reframe_opts = o
        return o

    def _reframe_worker(self, ids, o):
        failures = 0
        for n, mid in enumerate(ids):
            self._check()
            m = self.get_moment(mid)
            self._progress(f"Reframing clip {n + 1}/{len(ids)}…", n / len(ids))
            try:
                self._reframe_one(m, o, lambda f, n=n: self._progress(None, (n + f) / len(ids)))
            except Cancelled:
                raise
            except Exception as exc:
                failures += 1
                with self.lock:
                    m["reframe"] = {"status": "error", "error": str(exc)[:400]}
                self.bump()
        return f"Reframed {len(ids) - failures} of {len(ids)} clips." + (" See the cards for errors." if failures else "")

    def _reframe_one(self, m, o, progress):
        snap = self.snapshot
        vertical = o["aspect"] == "vertical"
        pieces = rm.video_pieces(snap, self.sel["video"], m["start"], m["end"])
        if not any(not p["gap"] for p in pieces):
            raise WorkflowError("There is no video on the chosen video tracks at this moment.")
        ap = rm.audio_pieces_by_track(snap, self.sel["audio"], m["start"], m["end"])
        swap = o["audio"] == "tracks" or (o["audio"] == "auto" and self._audio_differs(pieces, ap))
        out = self.work / "reframed" / f"m{m['id']}_{m['ver']}.mp4"
        warnings, method = [], None
        with self.lock:
            m["reframe"] = {"status": "running", "fraction": 0.0}
        self.bump()
        vcmd = self.tools.get("verthor")
        if vertical and o["smart"] and vcmd:
            device = {"gpu": "cuda", "cpu": "cpu"}.get(self.vgpu.get("status"))
            try:
                if len(pieces) == 1 and not pieces[0]["gap"]:
                    src, s0, s1 = pieces[0]["path"], pieces[0]["src_start"], pieces[0]["src_end"]
                else:
                    src = self.work / "src" / f"m{m['id']}_{m['ver']}.mp4"
                    rm.render_video_range(pieces, src, crf=12, preset="veryfast", cancel=self.cancel_event)
                    s0, s1 = 0.0, m["end"] - m["start"]
                raw = self.work / "reframed" / f"m{m['id']}_{m['ver']}_raw.mp4"
                # near-lossless intermediate (crf 14): the final quality is decided once, at export
                reframe_clip(vcmd, src, s0, s1, raw, o["preset"], o["fast"], crf=14, ffmpeg_preset="veryfast", device=device,
                             progress=lambda f: self._frac(m, f, progress), cancel=self.cancel_event)
                if swap:
                    rm.replace_audio(raw, ap, out, self.cancel_event)
                else:
                    shutil.move(str(raw), str(out))
                method = "verthor"
                if device == "cpu" and not o["fast"]:
                    warnings.append("verthor ran on the CPU. Enable the GPU for a big speed-up.")
            except ToolError as exc:
                warnings.append(f"verthor failed, so a centre crop was used instead. {str(exc)[-300:]}")
        elif vertical and o["smart"]:
            warnings.append("verthor is not installed, so a centre crop was used.")
        if method is None:
            rm.render_video_range(pieces, out, vertical_center=vertical, crf=14, preset="veryfast", audio_pieces=ap if swap else None, cancel=self.cancel_event)
            method = "center-crop" if vertical else "original"
        small = self.work / "reframed" / f"p{self.generation}_{m['id']}_{m['ver']}.mp4"
        self._small(out, small, 640 if vertical else 360)
        info = probe_video(out)
        with self.lock:
            m["reframe"] = {"status": "done", "method": method, "warnings": warnings, "file": f"reframed/{small.name}", "full": str(out),
                            "audio": "selected tracks" if swap else "camera", "aspect": "vertical" if vertical else "original",
                            "width": info.get("width"), "height": info.get("height")}
            m["captions"], m["cpreview"], m["final"] = {"status": "idle"}, {"status": "idle"}, {"status": "idle"}
        self.bump()

    def _frac(self, m, f, progress):
        with self.lock:
            if m["reframe"].get("status") == "running":
                m["reframe"]["fraction"] = f
        progress(f)

    def _small(self, src, dst, height):
        code, tail = run_cancelable(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vf", f"scale=-2:{height},setsar=1", "-c:v", "libx264",
                                     "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
                                     str(dst)], cancel=self.cancel_event)
        if code != 0:
            raise ToolError("Could not make the preview: " + " ".join(tail[-2:]))

    # ------------------------------------------------------------ 4. captions
    def _caption_options(self, opts):
        o = dict(self.caption_opts, **{k: v for k, v in (opts or {}).items() if k in DEFAULT_CAPTIONS})
        o["max_words"] = max(1, min(8, int(o["max_words"])))
        o["max_chars"] = max(0, min(80, int(o["max_chars"])))
        o["lines"] = 2 if int(o["lines"]) >= 2 else 1
        o["size"] = max(0.5, min(2.0, float(o["size"])))
        if o["density"] not in ("single", "less", "standard", "more", "custom"):
            o["density"] = "standard"
        if o["text_case"] not in ("none", "lowercase", "uppercase", "titlecase"):
            o["text_case"] = "none"
        if o["source"] not in ("autosubs", "builtin", "none"):
            o["source"] = "autosubs"
        self.caption_opts = o
        self._save_opts()
        return o

    def _style_and_format(self, o):
        style = cs.find_preset(self.presets, o["preset"])
        fmt = cs.Formatting(density=o["density"], max_words=o["max_words"], max_chars=o["max_chars"], lines=o["lines"], text_case=o["text_case"],
                            remove_punctuation=bool(o["remove_punctuation"]), size_scale=float(o["size"]), highlight=(o["highlight"] or "").lstrip("#").upper())
        return style, fmt

    def captions(self, ids, opts):
        picked = self.chosen(ids)
        missing = [m["id"] for m in picked if m["reframe"].get("status") != "done"]
        if missing:
            raise WorkflowError("Reframe these clips first (step 3): " + ", ".join(f"#{i}" for i in missing))
        o = self._caption_options(opts)
        with self.lock:
            for m in picked:
                m["captions"] = {"status": "queued"}
        self._run("captions", f"Transcribing {len(picked)} clip(s) for captions…", self._captions_worker, [m["id"] for m in picked], o)

    def _captions_worker(self, ids, o):
        failures = 0
        for n, mid in enumerate(ids):
            self._check()
            m = self.get_moment(mid)
            self._progress(f"Transcribing clip {n + 1}/{len(ids)}…", n / len(ids))
            try:
                self._captions_one(m, o)
            except Cancelled:
                raise
            except Exception as exc:
                failures += 1
                with self.lock:
                    m["captions"] = {"status": "error", "error": str(exc)[:400]}
                self.bump()
        return f"Captions ready for {len(ids) - failures} of {len(ids)} clips." + (" See the cards for errors." if failures else "")

    def _captions_one(self, m, o):
        """Word timings for one reframed clip. Style/formatting are applied later, so changing them never re-transcribes."""
        clip = Path(m["reframe"]["full"])
        warnings, words, engine = [], None, "none"
        with self.lock:
            m["captions"] = {"status": "running"}
        self.bump()
        if o["source"] == "autosubs" and self.tools.get("autosubs"):
            jpath = self.work / "captions" / f"m{m['id']}_{m['ver']}.json"
            try:
                if not (jpath.is_file() and jpath.stat().st_mtime >= clip.stat().st_mtime):
                    run_autosubs(self.tools["autosubs"], clip, jpath, o["autosubs_model"], self._lang(), cancel=self.cancel_event)
                words = cap.words_from_autosubs(json.loads(jpath.read_text(encoding="utf-8")))
                engine = "autosubs"
            except ToolError as exc:
                warnings.append(f"AutoSubs failed, so the built-in transcript was used. {str(exc)[-250:]}")
        elif o["source"] == "autosubs":
            warnings.append("AutoSubs is not installed, so the built-in transcript was used.")
        if o["source"] != "none" and not words:
            words, engine = cap.words_from_segments(self.transcript, m["start"], m["end"]), "builtin"
        with self.lock:
            m["_words"] = words or []
            m["captions"] = {"status": "done", "engine": engine, "warnings": warnings, "words": len(words or [])}
            m["cpreview"] = {"status": "idle"}
        self.bump()

    def _lang(self):
        lang = self.settings.get("language")
        return None if lang in (None, "", "auto") else lang

    def request_caption_preview(self, mid, opts=None):
        """Render the first seconds of a clip with the current caption style burned in (runs beside other jobs)."""
        m = self.get_moment(mid)
        if m["reframe"].get("status") != "done":
            raise WorkflowError("Reframe this clip first (step 3).")
        o = self._caption_options(opts)
        with self.lock:
            m["cpreview"] = {"status": "running"}
            stamp_id = self.rev + 1
        self.bump()
        self.previews.submit(self._caption_preview_worker, m["id"], m["ver"], o, stamp_id)

    def _caption_preview_worker(self, mid, ver, o, tag):
        try:
            m = self.get_moment(mid)
            if ver != m["ver"]:
                return
            if "_words" not in m and o["source"] != "none":
                self._captions_one(m, o)
            clip = Path(m["reframe"]["full"])
            style, fmt = self._style_and_format(o)
            info = probe_video(clip)
            words = [w for w in m.get("_words", []) if w["start"] < 8.0]
            work = self.work / "burn"
            base = f"cp{mid}_{tag}"
            seg = work / f"{base}.mp4"
            run_cancelable(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(clip), "-t", "8", "-vf", "scale=-2:720,setsar=1", "-c:v", "libx264",
                            "-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", str(seg)])
            scale = 720 / max(1, info.get("height") or 720)
            width, height = int(round((info.get("width") or 405) * scale)) // 2 * 2, 720
            font = cs.find_font(style.font, style.bold)
            ass = work / f"{base}.ass"
            ass.write_text(cs.build_ass(words, style, fmt, width, height, font), encoding="utf-8")
            out = self.work / "previews" / f"{base}.mp4"
            cs.burn_ass(seg, ass, out, fonts=[font["path"]], quality="fast")
            state = {"status": "ready", "file": f"previews/{out.name}", "font": font["family"] if font["found"] else f"{style.font} (not installed; Poppins used)"}
        except Exception as exc:
            state = {"status": "error", "error": str(exc)[:400]}
        with self.lock:
            try:
                self.get_moment(mid)["cpreview"] = state
            except WorkflowError:
                pass
        self.bump()

    def request_preset_thumbs(self):
        """Render a sample frame for every style, through the very same renderer the export uses."""
        for p in self.presets:
            if not p.get("thumb"):
                self.previews.submit(self._thumb_worker, p)

    def _thumb_worker(self, p):
        try:
            style = p["style"]
            font = cs.find_font(style.font, style.bold)
            work = self.work / "thumbs"
            tag = re.sub(r"[^A-Za-z0-9]+", "_", p["id"])
            base = work / f"{tag}.mp4"
            words = [{"word": w, "start": 0.05 + i * 0.3, "end": 0.3 + i * 0.3} for i, w in enumerate("GOD IS ABLE".split())]   # a full line, third word active
            run_cancelable(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "gradients=s=360x640:d=2:c0=0x1d2b64:c1=0xb36bd6:speed=0.01",
                            "-pix_fmt", "yuv420p", str(base)])
            ass = work / f"{tag}.ass"
            ass.write_text(cs.build_ass(words, style, cs.Formatting(), 360, 640, font), encoding="utf-8")
            vid = cs.burn_ass(base, ass, work / f"{tag}_out.mp4", fonts=[font["path"]], quality="fast")
            png = work / f"{tag}.png"
            run_cancelable(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", "0.82", "-i", str(vid), "-frames:v", "1", str(png)])
            with self.lock:
                p["thumb"] = f"thumbs/{png.name}"
                p["font_found"] = font["found"]
        except Exception as exc:
            with self.lock:
                p["thumb_error"] = str(exc)[:200]
        self.bump()

    def run_caption_test(self):
        def work():
            res = cs.selftest(work=self.work / "captest")
            with self.lock:
                self.captions_test = res
            self.bump()
        self.captions_test = {"running": True}
        self.bump()
        self.previews.submit(work)

    # ------------------------------------------------------------ 5. export
    def _use_nvenc(self):
        if self.export_opts.get("nvenc") == "off":
            return False
        if self._nvenc is None:
            self._nvenc = cs.nvenc_available()
        return self._nvenc

    def _export_options(self, opts):
        o = dict(self.export_opts, **{k: v for k, v in (opts or {}).items() if k in DEFAULT_EXPORT})
        if o["quality"] not in cs.QUALITY:
            o["quality"] = "high"
        self.export_opts = o
        self._save_opts()
        return o

    def _begin_export_dir(self, folder=None):
        self.export_dir = Path(folder) if folder else self.planned_export_dir()
        return self.export_dir

    def planned_export_dir(self):
        if self.custom_export_dir:
            return Path(self.custom_export_dir)
        return self.export_root / slug((self.snapshot or {}).get("timeline", "sermon"), 40)

    def export(self, ids, copts, eopts, folder=None):
        """Render the chosen clips. Needs them reframed; makes captions first if they don't exist yet."""
        picked = self.chosen(ids)
        missing = [m["id"] for m in picked if m["reframe"].get("status") != "done"]
        if missing:
            raise WorkflowError("Reframe these clips first (step 3), or use Export all: " + ", ".join(f"#{i}" for i in missing))
        co, eo = self._caption_options(copts), self._export_options(eopts)
        self._begin_export_dir(folder)
        with self.lock:
            for m in picked:
                m["final"] = {"status": "queued"}
        self._run("export", f"Exporting {len(picked)} clip(s)…", self._export_worker, [m["id"] for m in picked], co, eo, None)

    def export_all(self, ids, ropts, copts, eopts, folder=None):
        """The one-button path: reframe, caption and render every selected clip, skipping any stage already done."""
        if not self.sel["video"]:
            raise WorkflowError("Exporting needs a video track (step 1).")
        picked = self.chosen(ids)
        ro, co, eo = self._reframe_options(ropts), self._caption_options(copts), self._export_options(eopts)
        self._begin_export_dir(folder)
        with self.lock:
            for m in picked:
                if m["reframe"].get("status") != "done":
                    m["reframe"] = {"status": "queued"}
                m["final"] = {"status": "queued"}
        self._run("export_all", f"Exporting {len(picked)} clip(s)…", self._export_worker, [m["id"] for m in picked], co, eo, ro)

    def _export_worker(self, ids, co, eo, ro):
        out_dir = self.export_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"app": "Sermon-Snip", "version": VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "source_media": (self.snapshot or {}).get("timeline", ""), "output_dir": str(out_dir.resolve()), "clips": []}
        failures, total = 0, len(ids)
        for n, mid in enumerate(ids):
            self._check()
            m = self.get_moment(mid)

            def prog(stage, frac, n=n):
                weights = {"reframe": (0.0, 0.6), "captions": (0.6, 0.15), "render": (0.75, 0.25)}
                lo, w = weights[stage]
                self._progress(None, (n + lo + w * frac) / total)
            try:
                if ro is not None and m["reframe"].get("status") != "done":
                    self._progress(f"Clip {n + 1}/{total}: reframing…", (n) / total)
                    self._reframe_one(m, ro, lambda f: prog("reframe", f))
                if co["source"] != "none" and m["captions"].get("status") != "done":
                    self._progress(f"Clip {n + 1}/{total}: transcribing for captions…", (n + 0.6) / total)
                    self._captions_one(m, co)
                self._progress(f"Clip {n + 1}/{total}: rendering…", (n + 0.75) / total)
                manifest["clips"].append(self._export_one(m, co, eo, out_dir, n + 1, lambda f: prog("render", f)))
            except Cancelled:
                raise
            except Exception as exc:
                failures += 1
                with self.lock:
                    if m["reframe"].get("status") in ("queued", "running", "error") and ro is not None:
                        m["reframe"] = {"status": "error", "error": str(exc)[:400]}
                    m["final"] = {"status": "error", "error": str(exc)[:400]}
                self.bump()
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
            (HANDOFF_DIR / "last_manifest.txt").write_text(str((out_dir / "manifest.json").resolve()), encoding="utf-8")
        except OSError:
            pass
        with self.lock:
            self.last_manifest = str(out_dir / "manifest.json")
        return f"Exported {total - failures} of {total} clips to {out_dir}."

    def _export_one(self, m, co, eo, out_dir, rank, progress):
        clip = Path(m["reframe"]["full"])
        if not clip.is_file():
            raise WorkflowError("The reframed clip is missing; reframe it again.")
        info = probe_video(clip)
        warnings = list(m["reframe"].get("warnings", [])) + list(m["captions"].get("warnings", []))
        with self.lock:
            m["final"] = {"status": "running", "fraction": 0.0}
        self.bump()
        words = m.get("_words") if co["source"] != "none" else []
        base = f"{rank:02d}_{stamp(m['start']).replace(':', '-')}_{slug(m['title'])}"
        final = out_dir / f"{base}.mp4"
        style, fmt = self._style_and_format(co)
        srt_name = None
        burned = False
        if words:
            if eo["sidecar_srt"]:
                (out_dir / f"{base}.srt").write_text(cs.build_srt(words, cs.Formatting(density="more"), style), encoding="utf-8")
                srt_name = f"{base}.srt"
            font = cs.find_font(style.font, style.bold)
            if not font["found"]:
                warnings.append(f"The font '{style.font}' is not installed on this PC, so Poppins was used.")
            ass_text = cs.build_ass(words, style, fmt, info.get("width") or 1080, info.get("height") or 1920, font)
            if eo["sidecar_ass"]:
                (out_dir / f"{base}.ass").write_text(ass_text, encoding="utf-8")
            if co["burn"]:
                work_clip, work_ass = self.work / "burn" / f"{base}.mp4", self.work / "burn" / f"{base}.ass"
                shutil.copyfile(clip, work_clip)
                work_ass.write_text(ass_text, encoding="utf-8")
                progress(0.2)
                nv = self._use_nvenc()
                try:
                    cs.burn_ass(work_clip, work_ass, final, fonts=[font["path"]], quality=eo["quality"], nvenc=nv, cancel=self.cancel_event)
                except ToolError:
                    if not nv:
                        raise
                    cs.burn_ass(work_clip, work_ass, final, fonts=[font["path"]], quality=eo["quality"], nvenc=False, cancel=self.cancel_event)
                burned = True
        if not burned:
            shutil.copyfile(clip, final)         # no captions burned in: the reframed file is delivered as is (no re-encode)
        with self.lock:
            m["final"] = {"status": "done", "engine": m["captions"].get("engine", "none"), "warnings": warnings, "file": final.name, "captions": srt_name, "burned": burned}
        self.bump()
        return {"index": rank, "file": final.name, "captions": srt_name, "caption_engine": m["captions"].get("engine", "none"), "reframe": m["reframe"].get("method"),
                "vertical": m["reframe"].get("aspect") == "vertical", "width": info.get("width"), "height": info.get("height"), "fps": info.get("fps"),
                "duration": info.get("duration"), "source_start": m["start"], "source_end": m["end"], "title": m["title"], "score": m.get("score"),
                "text": m.get("text", ""), "warnings": warnings}

    def import_to_resolve(self):
        manifest = self.last_manifest
        if not manifest:
            raise WorkflowError("Export the clips first.")
        report = self.link.import_package(manifest)
        with self.lock:
            self.imported = report if report is not None else {"waiting": True}
        self.bump()

    # ------------------------------------------------------------ GPU helpers (Whisper/CUDA)
    def gpu_check(self):
        self._run("gpu-check", "Checking the NVIDIA GPU…", self._gpu_check_worker)

    def _gpu_check_worker(self):
        rep = gpu_setup.diagnose(prepare=True)
        if rep.ready:
            self._progress("Running a real GPU self-test (the first run downloads a 75 MB model)…", None)
            rep.checks.append(gpu_setup._selftest("tiny"))
        with self.lock:
            self.gpu.update({"status": "done", "ready": rep.ready, "text": rep.text()})
        return "GPU is ready." if rep.ready else "GPU is not ready. See the report."

    def gpu_install(self):
        self._run("gpu-install", "Installing the NVIDIA libraries (about 1.3 GB)…", self._gpu_install_worker)

    def _gpu_install_worker(self):
        gpu_setup.install_gpu_runtime(on_line=lambda line: self._progress(line[:140], None), cancel=self.cancel_event)
        return "NVIDIA libraries installed. Restart Sermon-Snip, then run the GPU check."

    # ------------------------------------------------------------ state for the UI
    def public_moment(self, m):
        keep = ("id", "start", "end", "score", "audio_score", "text_score", "peak_score", "signals", "text", "title", "selected", "ver", "ai_score", "ai_reason")
        d = {k: m[k] for k in keep if k in m}
        d["duration"] = round(m["end"] - m["start"], 2)
        d["orig"] = list(m["orig"])
        for key in ("preview", "reframe", "captions", "cpreview", "final"):
            d[key] = dict(m[key])
        d["reframe"].pop("full", None)
        return d

    def public_presets(self):
        return [{"id": p["id"], "name": p["name"], "description": p["description"], "source": p["source"], "thumb": p.get("thumb"),
                 "font": p["style"].font, "font_found": p.get("font_found")} for p in self.presets]

    def state(self):
        with self.lock:
            snap = self.snapshot
            return {
                "rev": self.rev, "version": VERSION, "generation": self.generation,
                "tools": {"ffmpeg": bool(self.tools.get("ffmpeg")), "verthor": bool(self.tools.get("verthor")), "autosubs": bool(self.tools.get("autosubs"))},
                "resolve": dict(self.link.status() if self.link else {}, source=(snap or {}).get("source")),
                "source": None if not snap else {
                    "kind": snap.get("source"), "timeline": snap.get("timeline"), "project": snap.get("project"), "fps": snap.get("fps"),
                    "duration": snap.get("duration"), "width": snap.get("width"), "height": snap.get("height"), "warnings": snap.get("warnings", []),
                    "resolve_version": snap.get("resolve_version", ""), "tracks": rm.track_summary(snap), "listening": getattr(self, "listening", False),
                    "suggested": rm.auto_select(snap)["audio"]},
                "selection": dict(self.sel), "settings": dict(self.settings), "reframe_opts": dict(self.reframe_opts),
                "caption_opts": dict(self.caption_opts), "export_opts": dict(self.export_opts), "quality": {k: v["label"] for k, v in cs.QUALITY.items()},
                "presets": self.public_presets(), "presets_info": dict(self.presets_info), "preset_names": PRESETS,
                "gemini": {"has_key": bool(gemini_rank.get_key(self.config)), "note": self.gemini_note, "default_model": gemini_rank.DEFAULT_MODEL},
                "verthor_gpu": dict(self.vgpu), "nvenc": self._nvenc, "captions_test": self.captions_test,
                "job": dict(self.job), "gpu": dict(self.gpu), "cut": dict(self.cut), "imported": self.imported,
                "language": self.language, "moments": [self.public_moment(m) for m in self.moments],
                "export_dir": str(self.export_dir or self.planned_export_dir()),
            }

    def media_roots(self):
        roots = [self.work]
        if self.export_dir:
            roots.append(self.export_dir)
        return roots

    def close(self):
        self.cancel_event.set()
        self.previews.shutdown(wait=False, cancel_futures=True)
