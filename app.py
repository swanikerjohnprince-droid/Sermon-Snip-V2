import csv
import json
import queue
import tempfile
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import gpu_setup
from analyzer import GpuError, analyze_wav_energy, candidates, transcribe
from autosubs_cli import transcribe_autosubs
from media import cache_file, ffmpeg_available, prepare_audio, stamp
from pipeline import ShortsOptions, build_shorts, write_pointer
from reframe import PRESETS
from scene_detect import detect_cuts, has_video, make_scenes, probe_duration
from tools import Cancelled, find_autosubs, find_verthor, load_config, save_config

VERSION = "0.5"
MEDIA_TYPES = [("Audio or video", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.mp4 *.mov *.mkv *.mxf *.avi *.webm"),
               ("All files", "*.*")]
WHISPER_MODELS = ["medium", "distil-large-v3", "large-v3", "small", "base", "tiny"]
AUTOSUBS_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"]
ENGINES = ["Whisper (faster-whisper)", "AutoSubs (no CUDA needed)"]
DEVICES = ["GPU (NVIDIA CUDA)", "CPU"]
LANGUAGES = ["en", "auto", "es", "fr", "pt", "de", "sw", "yo", "ha"]
CAPTION_MODES = ["AutoSubs", "Built-in", "None"]
LOG_FILE = Path(__file__).resolve().parent / "sermon_snip_error.log"


def log_error(label, details):
    try:
        with LOG_FILE.open("a", encoding="utf-8") as log:
            log.write("\n" + "=" * 70 + f"\n{label}\n{details}\n")
    except OSError:
        pass

class App:
    def __init__(self, root):
        self.root = root
        root.title("Sermon-Snip — Sermon Highlight Finder")
        root.geometry("1180x860")
        root.minsize(980, 680)
        self.config = load_config()
        self.audio = None            # sermon audio or video (anything FFmpeg can read)
        self.video = None            # optional separate video for scene cuts / clip export
        self.results, self.transcript, self.energy = [], [], []
        self.duration, self.language = 0, "unknown"
        self.busy = False
        self.cancel = threading.Event()
        self.events = queue.Queue()
        self.action_buttons = []
        c = self.config
        self.engine = tk.StringVar(value=c.get("engine", ENGINES[0]))
        self.device = tk.StringVar(value=c.get("device", DEVICES[0]))
        self.model = tk.StringVar(value=c.get("model", "large-v3"))
        self.lang = tk.StringVar(value=c.get("language", "en"))
        self.vocabulary = tk.StringVar(value="")
        self.target = tk.IntVar(value=c.get("target_seconds", 60))
        self.smart_reframe = tk.BooleanVar(value=True)
        self.preset = tk.StringVar(value="talking_head")
        self.fast = tk.BooleanVar(value=False)
        self.caption_mode = tk.StringVar(value="AutoSubs")
        self.burn = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Choose a sermon recording (audio or video) to begin.")
        self.tools_text = tk.StringVar()
        self.build_ui()
        self.refresh_tools()
        root.after(150, self.poll)

    # ------------------------------------------------------------ UI
    def btn(self, parent, text, command, action=True):
        button = ttk.Button(parent, text=text, command=command)
        if action:
            self.action_buttons.append(button)
        return button

    def build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#10141b")
        style.configure("TLabel", background="#10141b", foreground="#e9eef5")
        style.configure("TCheckbutton", background="#10141b", foreground="#e9eef5")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), foreground="#ffffff")
        style.configure("Sub.TLabel", foreground="#aeb9c8")
        style.configure("Treeview", rowheight=32, background="#171e28", foreground="#e9eef5",
                        fieldbackground="#171e28")
        style.configure("Treeview.Heading", background="#273241", foreground="#ffffff")
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Sermon-Snip", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Find sermon highlights • reframe to vertical • caption • send to DaVinci Resolve",
                  style="Sub.TLabel").pack(anchor="w", pady=(2, 10))

        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=3)
        self.btn(bar, "Choose sermon audio/video", self.choose_audio, action=False).pack(side="left")
        self.btn(bar, "Choose video for cuts/clips", self.choose_video, action=False).pack(side="left", padx=6)
        ttk.Label(bar, text="Model").pack(side="left", padx=(8, 3))
        self.model_box = ttk.Combobox(bar, textvariable=self.model, values=WHISPER_MODELS, width=17)
        self.model_box.pack(side="left")
        ttk.Label(bar, text="Sermon vocabulary (optional)").pack(side="left", padx=(8, 3))
        ttk.Entry(bar, textvariable=self.vocabulary, width=26).pack(side="left")

        bar2 = ttk.Frame(outer)
        bar2.pack(fill="x", pady=3)
        ttk.Label(bar2, text="Engine").pack(side="left", padx=(0, 3))
        engine_box = ttk.Combobox(bar2, textvariable=self.engine, values=ENGINES, width=24, state="readonly")
        engine_box.pack(side="left")
        engine_box.bind("<<ComboboxSelected>>", self.on_engine_change)
        ttk.Label(bar2, text="Device").pack(side="left", padx=(8, 3))
        ttk.Combobox(bar2, textvariable=self.device, values=DEVICES, width=17, state="readonly").pack(side="left")
        ttk.Label(bar2, text="Language").pack(side="left", padx=(8, 3))
        ttk.Combobox(bar2, textvariable=self.lang, values=LANGUAGES, width=6).pack(side="left")
        ttk.Label(bar2, text="Clip length (s)").pack(side="left", padx=(8, 3))
        ttk.Spinbox(bar2, textvariable=self.target, from_=15, to=120, width=5).pack(side="left")

        bar3 = ttk.Frame(outer)
        bar3.pack(fill="x", pady=3)
        self.btn(bar3, "Analyze", self.analyze).pack(side="left")
        self.btn(bar3, "Re-rank", self.rerank).pack(side="left", padx=6)
        self.btn(bar3, "Detect cuts", self.detect).pack(side="left")
        self.btn(bar3, "Check GPU", self.check_gpu).pack(side="left", padx=6)
        self.cancel_button = ttk.Button(bar3, text="Cancel", command=self.request_cancel, state="disabled")
        self.cancel_button.pack(side="left")

        self.audio_label = ttk.Label(outer, text="Sermon file: none", style="Sub.TLabel")
        self.audio_label.pack(anchor="w", pady=1)
        self.video_label = ttk.Label(outer, text="Video: uses the sermon file if it has video", style="Sub.TLabel")
        self.video_label.pack(anchor="w", pady=1)
        ttk.Label(outer, textvariable=self.tools_text, style="Sub.TLabel").pack(anchor="w", pady=1)
        ttk.Label(outer, text="Candidate highlights — highest score first; hover over an excerpt to read it all",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 5))
        self.tree = ttk.Treeview(outer, columns=("score", "start", "end", "text"), show="headings",
                                 selectmode="extended")
        for col, label, width in [("score", "Score", 65), ("start", "Start", 80),
                                  ("end", "End", 80), ("text", "Transcript excerpt", 700)]:
            self.tree.heading(col, text=label)
            self.tree.column(col, width=width, anchor="w" if col == "text" else "center")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.detail)
        self.tree.bind("<Motion>", self.show_excerpt_tooltip)
        self.tree.bind("<Leave>", self.hide_excerpt_tooltip)
        self.tooltip = None
        self.tooltip_row = None

        row = ttk.Frame(outer)
        row.pack(fill="x", pady=(8, 2))
        self.btn(row, "Export highlights JSON", self.export_json).pack(side="left")
        self.btn(row, "Export highlights CSV", self.export_csv).pack(side="left", padx=8)
        ttk.Label(row, text="Select clips with Ctrl/Shift-click; none selected = all",
                  style="Sub.TLabel").pack(side="left", padx=8)

        shorts = ttk.Frame(outer)
        shorts.pack(fill="x", pady=2)
        ttk.Checkbutton(shorts, text="Smart reframe 9:16 (verthor)", variable=self.smart_reframe).pack(side="left")
        ttk.Combobox(shorts, textvariable=self.preset, values=PRESETS, width=13, state="readonly").pack(side="left", padx=4)
        ttk.Checkbutton(shorts, text="Fast", variable=self.fast).pack(side="left")
        ttk.Label(shorts, text="Captions").pack(side="left", padx=(12, 3))
        ttk.Combobox(shorts, textvariable=self.caption_mode, values=CAPTION_MODES, width=9,
                     state="readonly").pack(side="left")
        ttk.Checkbutton(shorts, text="Burn in", variable=self.burn).pack(side="left", padx=4)
        self.btn(shorts, "Build shorts package…", self.build_package).pack(side="left", padx=8)
        self.btn(shorts, "Locate tools…", self.locate_tools, action=False).pack(side="left")

        self.details = tk.Text(outer, height=4, wrap="word", bg="#171e28", fg="#e9eef5",
                               relief="flat", padx=8, pady=6)
        self.details.pack(fill="x", pady=(6, 0))
        self.details.insert("1.0", "Select a result to inspect transcript and scoring signals.")
        self.details.configure(state="disabled")
        self.bar = ttk.Progressbar(outer, mode="determinate", maximum=100)
        self.bar.pack(fill="x", pady=(8, 0))
        ttk.Label(outer, textvariable=self.status, style="Sub.TLabel").pack(anchor="w", pady=(5, 0))

    def show_excerpt_tooltip(self, event):
        # Show the complete transcript excerpt when hovering over its table cell.
        row_id = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if not row_id or column != "#4":
            self.hide_excerpt_tooltip()
            return
        if row_id == self.tooltip_row:
            return
        self.hide_excerpt_tooltip()
        try:
            excerpt = self.tree.set(row_id, "text")
            if not excerpt:
                return
            popup = tk.Toplevel(self.root)
            popup.wm_overrideredirect(True)
            popup.attributes("-topmost", True)
            x = min(event.x_root + 18, self.root.winfo_screenwidth() - 600)
            y = min(event.y_root + 18, self.root.winfo_screenheight() - 300)
            popup.geometry(f"+{max(0, x)}+{max(0, y)}")
            frame = tk.Frame(popup, bg="#202a38", bd=1, relief="solid")
            frame.pack(fill="both", expand=True)
            tk.Label(
                frame, text="FULL EXCERPT", anchor="w",
                bg="#202a38", fg="#9fc5ff", font=("Segoe UI", 9, "bold"),
                padx=10, pady=6
            ).pack(fill="x")
            body = tk.Text(
                frame, wrap="word", width=72, height=12,
                bg="#171e28", fg="#e9eef5", relief="flat",
                padx=10, pady=8, font=("Segoe UI", 10)
            )
            scroll = ttk.Scrollbar(frame, orient="vertical", command=body.yview)
            body.configure(yscrollcommand=scroll.set)
            body.insert("1.0", excerpt)
            body.configure(state="disabled")
            body.pack(side="left", fill="both", expand=True)
            scroll.pack(side="right", fill="y")
            self.tooltip = popup
            self.tooltip_row = row_id
        except tk.TclError:
            self.tooltip = None
            self.tooltip_row = None

    def hide_excerpt_tooltip(self, _event=None):
        if self.tooltip is not None:
            try:
                self.tooltip.destroy()
            except tk.TclError:
                pass
        self.tooltip = None
        self.tooltip_row = None

    # ------------------------------------------------------------ choices & settings
    def choose_audio(self):
        path = filedialog.askopenfilename(filetypes=MEDIA_TYPES)
        if path:
            self.audio = Path(path)
            self.audio_label.config(text="Sermon file: " + path)
            self.transcript, self.energy, self.results = [], [], []
            self.fill_tree()
            if not ffmpeg_available():
                self.status.set("FFmpeg not found: only 16-bit PCM WAV will work, and clip export is unavailable.")

    def choose_video(self):
        path = filedialog.askopenfilename(filetypes=MEDIA_TYPES)
        if path:
            self.video = Path(path)
            self.video_label.config(text="Video: " + path)

    def video_source(self):
        return self.video or self.audio

    def on_engine_change(self, _=None):
        autosubs = self.engine.get() == ENGINES[1]
        self.model_box.configure(values=AUTOSUBS_MODELS if autosubs else WHISPER_MODELS)
        if self.model.get() not in (AUTOSUBS_MODELS if autosubs else WHISPER_MODELS):
            self.model.set("large-v3-turbo" if autosubs else "large-v3")

    def refresh_tools(self):
        self.verthor_cmd, self.autosubs_cmd = find_verthor(self.config), find_autosubs(self.config)
        self.tools_text.set(
            "Tools — FFmpeg: " + ("found" if ffmpeg_available() else "NOT found") +
            " | verthor: " + ("found" if self.verthor_cmd else "not found (centre-crop fallback)") +
            " | AutoSubs: " + ("found" if self.autosubs_cmd else "not found (built-in captions)"))

    def locate_tools(self):
        for key, label in (("autosubs_path", "AutoSubs (AutoSubs.exe / autosubs)"),
                           ("verthor_path", "verthor (verthor.exe / verthor)")):
            if messagebox.askyesno("Locate tools", f"Browse for {label}?\n\nChoose No to skip."):
                path = filedialog.askopenfilename(title=f"Locate {label}")
                if path:
                    self.config[key] = path
        save_config(self.config)
        self.refresh_tools()

    def remember(self, **values):
        self.config.update(values)
        try:
            save_config(self.config)
        except OSError:
            pass

    # ------------------------------------------------------------ task plumbing
    def start_task(self, target, *args):
        if self.busy:
            return
        self.busy = True
        self.cancel.clear()
        for button in self.action_buttons:
            button.state(["disabled"])
        self.cancel_button.state(["!disabled"])
        self.events.put(("progress", None))
        threading.Thread(target=self.guarded, args=(target, args), daemon=True).start()

    def guarded(self, target, args):
        try:
            target(*args)
        except Cancelled:
            self.events.put(("status", "Cancelled."))
        except GpuError as exc:
            log_error("GPU failure", f"{exc}\n\n{exc.diagnosis}")
            self.events.put(("gpu_error", exc))
        except Exception as exc:
            details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            log_error(f"{target.__name__} failed", details)
            self.events.put(("error", f"{type(exc).__name__}: {exc}\n\nFull traceback saved to "
                                      f"{LOG_FILE.name}.\n\n{details}"))
        finally:
            self.events.put(("idle", None))

    def report(self, message, fraction=None):
        self.events.put(("status", message))
        self.events.put(("progress", fraction))

    def request_cancel(self):
        self.cancel.set()
        self.status.set("Cancelling…")

    # ------------------------------------------------------------ analysis
    def analyze(self):
        if not self.audio:
            messagebox.showinfo("Choose a file", "Choose a sermon audio or video file first.")
            return
        try:
            target = int(self.target.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Settings", "Clip length must be a whole number of seconds.")
            return
        autosubs = self.engine.get() == ENGINES[1]
        if autosubs and not self.autosubs_cmd:
            messagebox.showerror("AutoSubs not found", "Install AutoSubs (github.com/tmoroney/auto-subs) or use "
                                 "'Locate tools…' to point to it.")
            return
        opts = {"model": self.model.get(), "vocabulary": self.vocabulary.get().strip(), "autosubs": autosubs,
                "cpu": self.device.get() == DEVICES[1], "target": target,
                "language": None if self.lang.get().strip().lower() in ("", "auto") else self.lang.get().strip().lower()}
        self.remember(engine=self.engine.get(), device=self.device.get(), model=opts["model"],
                      language=self.lang.get(), target_seconds=target)
        self.status.set("Analyzing audio and transcribing…")
        self.start_task(self.worker, opts)

    def worker(self, opts):
        with tempfile.TemporaryDirectory(prefix="sermonsnip_") as tmp:
            wav = prepare_audio(self.audio, tmp, self.report)
            duration, energy = analyze_wav_energy(
                wav, progress=lambda f: self.report(f"Analyzing audio energy… {f:.0%}", f))
            transcript, language = self.get_transcript(wav, opts)
        self.report("Ranking highlights…")
        results = candidates(transcript, energy, duration, target_seconds=opts["target"])
        self.events.put(("done", (results, transcript, language, duration, energy)))

    def get_transcript(self, wav, opts):
        """Transcribe with the chosen engine; results are cached so re-analysing is instant."""
        tag = ("autosubs-" if opts["autosubs"] else "whisper-") + opts["model"]
        cache = cache_file(self.audio, tag, opts["language"], extra=opts["vocabulary"])
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            self.report("Using cached transcript for this file and settings.", 1.0)
            return data["segments"], data.get("language", "unknown")
        except (OSError, ValueError, KeyError):
            pass
        if opts["autosubs"]:
            extra = (["--no-gpu"] if opts["cpu"] else []) + (["--prompt", opts["vocabulary"]] if opts["vocabulary"] else [])
            rows, language = transcribe_autosubs(self.autosubs_cmd, wav, opts["model"], opts["language"],
                                                 self.report, cache_id=self.audio, cancel=self.cancel, extra=extra)
        else:
            rows, language = transcribe(
                wav, opts["model"], progress=lambda m: self.report(m), language=opts["language"],
                device="cpu" if opts["cpu"] else "cuda", chunk_seconds=30, overlap_seconds=2,
                initial_prompt=opts["vocabulary"] or None)
        try:
            cache.parent.mkdir(exist_ok=True)
            cache.write_text(json.dumps({"language": language, "segments": rows}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return rows, language

    def rerank(self):
        if not self.transcript:
            messagebox.showinfo("Nothing to re-rank", "Run Analyze first.")
            return
        try:
            target = int(self.target.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Settings", "Clip length must be a whole number of seconds.")
            return
        self.start_task(self.rerank_worker, target)

    def rerank_worker(self, target):
        self.report("Ranking highlights…")
        results = candidates(self.transcript, self.energy, self.duration, target_seconds=target)
        self.events.put(("done", (results, self.transcript, self.language, self.duration, self.energy)))

    # ------------------------------------------------------------ GPU help
    def check_gpu(self):
        self.status.set("Checking the NVIDIA GPU setup…")
        self.start_task(self.gpu_check_worker)

    def gpu_check_worker(self):
        self.report("Checking drivers and CUDA libraries…")
        rep = gpu_setup.diagnose(prepare=True)
        if rep.ready:
            self.report("Running a real GPU self-test (first run downloads a 75 MB model)…")
            rep.checks.append(gpu_setup._selftest("tiny"))
        self.events.put(("gpu_report", rep))

    def install_gpu_worker(self):
        self.report("Installing NVIDIA runtime libraries with pip (about 1.3 GB; this can take a while)…")
        gpu_setup.install_gpu_runtime(on_line=lambda line: self.events.put(("status", line[:140])),
                                      cancel=self.cancel)
        self.events.put(("gpu_installed", None))

    def show_gpu_dialog(self, title, headline, body, recover=False):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("860x560")
        win.configure(background="#10141b")
        ttk.Label(win, text=headline, font=("Segoe UI", 12, "bold"), wraplength=820).pack(anchor="w", padx=14, pady=(12, 6))
        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=14)
        text = tk.Text(frame, wrap="word", bg="#171e28", fg="#e9eef5", relief="flat", padx=8, pady=6,
                       font=("Consolas", 10))
        scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.insert("1.0", body)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        buttons = ttk.Frame(win)
        buttons.pack(fill="x", padx=14, pady=10)

        def copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(body)

        def install():
            win.destroy()
            if messagebox.askyesno("Install GPU libraries",
                                   "This downloads about 1.3 GB (NVIDIA cuBLAS 12 and cuDNN 9) into this app's "
                                   "Python environment. It does not change your system CUDA.\n\nContinue?"):
                self.start_task(self.install_gpu_worker)

        def use_cpu():
            win.destroy()
            self.device.set(DEVICES[1])
            self.analyze()

        def use_autosubs():
            win.destroy()
            self.engine.set(ENGINES[1])
            self.on_engine_change()
            self.analyze()

        ttk.Button(buttons, text="Copy report", command=copy).pack(side="left")
        ttk.Button(buttons, text="Install GPU libraries (≈1.3 GB)", command=install).pack(side="left", padx=6)
        if recover:
            ttk.Button(buttons, text="Run on CPU this time", command=use_cpu).pack(side="left")
            if self.autosubs_cmd:
                ttk.Button(buttons, text="Use AutoSubs engine instead", command=use_autosubs).pack(side="left", padx=6)
        ttk.Button(buttons, text="Close", command=win.destroy).pack(side="right")

    # ------------------------------------------------------------ scene cuts
    def detect(self):
        if not self.video_source():
            messagebox.showinfo("Choose video", "Choose a video first.")
            return
        self.status.set("Detecting likely visual cuts with FFmpeg…")
        self.start_task(self.scene_worker, self.video_source())

    def scene_worker(self, video):
        if not has_video(video):
            raise RuntimeError("That file has no video stream. Use 'Choose video for cuts/clips'.")
        duration = probe_duration(video)
        cuts = detect_cuts(video)
        self.events.put(("scenes", (video, duration, cuts, make_scenes(cuts, duration))))

    # ------------------------------------------------------------ shorts package
    def build_package(self):
        if not self.results:
            messagebox.showinfo("No results", "Analyze audio first.")
            return
        video = self.video_source()
        if not video or not has_video(video):
            messagebox.showinfo("Video needed", "Building shorts needs a video file. Choose one with "
                                "'Choose video for cuts/clips' (it must share the audio's timeline start).")
            return
        picked = sorted((self.results[int(i)] for i in self.tree.selection()), key=lambda r: r["start"]) \
            or sorted(self.results, key=lambda r: r["start"])
        outdir = filedialog.askdirectory(title=f"Folder for {len(picked)} short(s)")
        if not outdir:
            return
        mode = self.caption_mode.get()
        opts = ShortsOptions(
            vertical=True, smart_reframe=self.smart_reframe.get(), preset=self.preset.get(), fast=self.fast.get(),
            captions={"AutoSubs": "autosubs", "Built-in": "builtin", "None": "none"}[mode],
            burn_captions=self.burn.get(), verthor_cmd=self.verthor_cmd, autosubs_cmd=self.autosubs_cmd,
            language=None if self.lang.get().strip().lower() in ("", "auto") else self.lang.get().strip().lower())
        self.status.set(f"Building {len(picked)} short(s)…")
        self.start_task(self.package_worker, video, picked, outdir, opts)

    def package_worker(self, video, picked, outdir, opts):
        manifest = build_shorts(video, picked, self.transcript, outdir, opts, progress=self.report, cancel=self.cancel)
        self.events.put(("package", (manifest, outdir)))

    # ------------------------------------------------------------ event loop
    def poll(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "status":
                    self.status.set(data)
                elif kind == "progress":
                    self.set_progress(data)
                elif kind == "idle":
                    self.busy = False
                    self.bar.stop()
                    self.bar.configure(mode="determinate", value=0)
                    for button in self.action_buttons:
                        button.state(["!disabled"])
                    self.cancel_button.state(["disabled"])
                elif kind == "error":
                    self.status.set("Task failed — see error details.")
                    messagebox.showerror("Sermon-Snip — detailed error", data if data.strip() else "Unknown error.")
                elif kind == "gpu_error":
                    self.status.set("GPU transcription could not start — see the report.")
                    self.show_gpu_dialog("Sermon-Snip — GPU problem", "NVIDIA GPU transcription failed. Nothing was "
                                         "switched to CPU automatically.",
                                         f"{data}\n\n--- Diagnosis ---\n{data.diagnosis}", recover=True)
                elif kind == "gpu_report":
                    self.status.set("GPU check complete.")
                    self.show_gpu_dialog("Sermon-Snip — GPU check", "GPU is ready." if data.ready else
                                         "GPU is NOT ready.", data.text())
                elif kind == "gpu_installed":
                    self.status.set("NVIDIA libraries installed. Restart Sermon-Snip, then click Check GPU.")
                    messagebox.showinfo("Restart needed", "NVIDIA libraries installed.\n\nClose and reopen Sermon-Snip, "
                                        "then click Check GPU to confirm.")
                elif kind == "done":
                    self.results, self.transcript, self.language, self.duration, self.energy = data
                    self.results.sort(key=lambda item: item.get("score", 0), reverse=True)
                    self.fill_tree()
                    self.status.set(f"Analysis complete: {len(self.results)} candidates; language {self.language}.")
                elif kind == "scenes":
                    self.save_scenes(*data)
                elif kind == "package":
                    manifest, outdir = data
                    warned = [c for c in manifest["clips"] if c["warnings"]]
                    self.status.set(f"Built {len(manifest['clips'])} short(s) in {outdir}")
                    msg = (f"Built {len(manifest['clips'])} short(s).\n\nNext: in DaVinci Resolve run Workspace → Scripts → "
                           "SermonSnip_Import, then style captions with AutoSubs (Import → the .srt files).")
                    if warned:
                        msg += "\n\nFallbacks used:\n" + "\n".join(f"• clip {c['index']}: {c['warnings'][0][:150]}" for c in warned[:6])
                    messagebox.showinfo("Shorts package ready", msg)
        except queue.Empty:
            pass
        self.root.after(150, self.poll)

    def set_progress(self, fraction):
        if fraction is None:
            self.bar.configure(mode="indeterminate")
            self.bar.start(12)
        else:
            self.bar.stop()
            self.bar.configure(mode="determinate", value=max(0, min(1, fraction)) * 100)

    def fill_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, result in enumerate(self.results):
            self.tree.insert("", "end", iid=str(i), values=(
                result["score"], stamp(result["start"]), stamp(result["end"]), result["text"]))

    def detail(self, _=None):
        selection = self.tree.selection()
        if not selection:
            return
        result = self.results[int(selection[0])]
        extra = f" (+{len(selection) - 1} more selected)" if len(selection) > 1 else ""
        text = (f"Score {result['score']}/100 | Audio {result['audio_score']:.2f} | "
                f"Text {result['text_score']:.2f} | Peak {result['peak_score']:.2f}{extra}\n"
                f"Signals: {', '.join(result['signals']) or 'No configured phrase signals'}\n\n{result['text']}")
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    # ------------------------------------------------------------ exports
    def save_scenes(self, video, duration, cuts, scenes):
        output = filedialog.asksaveasfilename(defaultextension=".json", initialfile="sermon_snip_scenes.json",
                                              filetypes=[("JSON", "*.json")])
        if output:
            Path(output).write_text(json.dumps({
                "app": "Sermon-Snip", "version": VERSION, "source_video": str(video),
                "duration_seconds": duration, "scene_cut_threshold": .32, "cut_times_seconds": cuts,
                "scenes": scenes, "note": "Scene detection only."}, indent=2), encoding="utf-8")
            self.status.set(f"Detected {len(cuts)} cuts across {len(scenes)} scenes. Report saved.")

    def export_json(self):
        if not self.results:
            messagebox.showinfo("No results", "Analyze audio first.")
            return
        output = filedialog.asksaveasfilename(defaultextension=".json", initialfile="sermon_snip_highlights.json",
                                              filetypes=[("JSON", "*.json")])
        if output:
            Path(output).write_text(json.dumps({
                "app": "Sermon-Snip", "version": VERSION, "source_audio": str(self.audio),
                "duration_seconds": self.duration, "language": self.language,
                "timing_note": "Times are relative to source audio start. Verify offsets in Resolve.",
                "highlights": self.results}, indent=2, ensure_ascii=False), encoding="utf-8")
            write_pointer("last_highlights.txt", str(Path(output).resolve()))
            self.status.set("Exported " + output)

    def export_csv(self):
        if not self.results:
            messagebox.showinfo("No results", "Analyze audio first.")
            return
        output = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="sermon_snip_highlights.csv",
                                              filetypes=[("CSV", "*.csv")])
        if output:
            with open(output, "w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "start", "end", "score", "audio_score", "text_score", "peak_score", "signals", "text"],
                    extrasaction="ignore")
                writer.writeheader()
                for item in self.results:
                    row = dict(item)
                    row["signals"] = "; ".join(item["signals"])
                    writer.writerow(row)
            self.status.set("Exported " + output)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
