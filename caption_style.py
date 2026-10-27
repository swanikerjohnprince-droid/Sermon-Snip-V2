"""Caption styles for Sermon-Snip: its own styles + AutoSubs presets (built-in and the user's own), rendered as ASS.

AutoSubs stores caption presets as raw values of a Fusion Text+ macro (font, size, position, fill/outline/shadow,
highlight Fill/Outline/Glow/Bubble, pop-in/fade/slide). ``style_from_fusion`` maps those values onto a renderer that
burns the captions into the exported video, so a preset looks like what you designed in AutoSubs. It is a close
reproduction, not pixel-identical: Fusion's text engine is only available inside Resolve.

Fonts: burning never depends on what is installed. The needed font file (your preset's font if this PC has it,
else the bundled Poppins, OFL-licensed) is copied beside the subtitle file and handed to libass with ``fontsdir=.``.
"""
import glob
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
BUNDLED_FONTS = APP_DIR / "fonts"
BUILTIN_PRESETS_FILE = APP_DIR / "data" / "autosubs_builtin_presets.json"
HIGHLIGHT_STYLES = {0: "fill", 1: "outline", 2: "glow", 3: "bubble"}   # the Fusion macro's HighlightStyle combo
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ---------------------------------------------------------------- fonts

def _read_font_names(path):
    """{nameID: text} from a TrueType/OpenType name table (nameID 1 family, 2 style, 16/17 typographic)."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if head[:4] == b"ttcf":
                return {}
            n = struct.unpack(">H", head[4:6])[0]
            table = f.read(16 * n)
            for i in range(n):
                tag, _, off, ln = struct.unpack(">4sIII", table[16 * i:16 * i + 16])
                if tag == b"name":
                    f.seek(off)
                    data = f.read(ln)
                    break
            else:
                return {}
        count, string_off = struct.unpack(">HH", data[2:6])
        out = {}
        for j in range(count):
            pid, _eid, _lid, nid, length, offset = struct.unpack(">HHHHHH", data[6 + 12 * j:18 + 12 * j])
            if nid in (1, 2, 16, 17):
                raw = data[string_off + offset:string_off + offset + length]
                text = raw.decode("utf-16-be", "replace") if pid in (0, 3) else raw.decode("latin-1", "replace")
                out.setdefault(nid, text) if pid != 3 else out.__setitem__(nid, text)
        return out
    except (OSError, struct.error):
        return {}


def system_font_dirs():
    if sys.platform == "win32":
        return [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
                Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Microsoft" / "Windows" / "Fonts"]
    if sys.platform == "darwin":
        return [Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library" / "Fonts"]
    return [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts", Path.home() / ".local/share/fonts"]


_FONT_INDEX = None


def font_index(dirs=None, refresh=False):
    """[(family_lower, style_lower, family, path)] for installed fonts and the bundled ones. Scanned once."""
    global _FONT_INDEX
    if _FONT_INDEX is not None and not refresh and dirs is None:
        return _FONT_INDEX
    entries = []
    for root in ([BUNDLED_FONTS] + (system_font_dirs() if dirs is None else [Path(d) for d in dirs])):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() in (".ttf", ".otf"):
                names = _read_font_names(path)
                fam = names.get(16) or names.get(1)
                if fam:
                    # nameID 1 is what libass matches; the typographic family (16) is what designers type
                    entries.append((fam.lower(), (names.get(17) or names.get(2) or "").lower(), names.get(1) or fam, path, (names.get(2) or "").lower()))
    if dirs is None:
        _FONT_INDEX = entries
    return entries


def find_font(family, bold=False, index=None):
    """Locate a font file for a family name. Returns {'family','bold','path','found'}; falls back to bundled Poppins."""
    want = (family or "").strip().lower()
    entries = index if index is not None else font_index()
    candidates = ([e for e in entries if e[2].lower() == want]            # the exact family name libass will match
                  or [e for e in entries if e[0] == want]                  # the designer-facing (typographic) family
                  or [e for e in entries if want and e[0] and (want in e[0] or e[0] in want)])
    if candidates:
        def rank(e):
            is_bold = "bold" in e[1] or "extrabold" in e[1].replace(" ", "")
            return (is_bold != bold, "italic" in e[1] or "oblique" in e[1], e[3].parent != BUNDLED_FONTS)
        best = sorted(candidates, key=rank)[0]
        flag = bold and "bold" not in best[4]          # ask libass to embolden only if the file's own style is not already bold
        return {"family": best[2], "bold": flag, "path": best[3], "found": True}
    fallback = BUNDLED_FONTS / ("Poppins-Bold.ttf")
    return {"family": "Poppins", "bold": False, "path": fallback, "found": False}


# ---------------------------------------------------------------- style model

@dataclass
class CaptionStyle:
    id: str = "ss:bold_pop"
    name: str = "Bold pop"
    source: str = "sermonsnip"                 # sermonsnip | autosubs | autosubs-user
    description: str = ""
    font: str = "Poppins"
    bold: bool = True
    size: float = 0.055                        # fraction of video height
    fill: str = "FFFFFF"
    outline_on: bool = True
    outline: str = "000000"
    outline_w: float = 0.12                    # Fusion "thickness": fraction of the font size
    shadow_on: bool = False
    shadow: str = "000000"
    highlight_on: bool = True
    highlight: str = "FFD600"
    highlight_style: str = "fill"              # fill | outline | glow | bubble
    pos_x: float = 0.5
    pos_y: float = 0.15                        # from the bottom edge, like Fusion
    uppercase: bool = False
    pop_in: bool = False
    fade: bool = False
    slide_up: bool = False
    anim_len: float = 0.2
    reveal: str = "all"                        # all | progressive (words appear as spoken)

    def to_dict(self):
        return asdict(self)


@dataclass
class Formatting:
    """AutoSubs' caption-formatting options (density, lines, words/characters per line, case, punctuation)."""
    density: str = "standard"                  # single | less | standard | more | custom
    max_words: int = 3                         # used by 'custom'
    max_chars: int = 0                         # 0 = no limit
    lines: int = 1
    text_case: str = "none"                    # none | lowercase | uppercase | titlecase
    remove_punctuation: bool = False
    size_scale: float = 1.0                    # extra text-size multiplier from the UI
    highlight: str = ""                        # RGB hex override for the highlight colour ("" = style's own)

    @property
    def words_per_line(self):
        return {"single": 1, "less": 2, "standard": 3, "more": 5}.get(self.density, max(1, int(self.max_words)))


def _hex(r, g, b):
    return "".join(f"{max(0, min(255, int(round(float(v) * 255)))):02X}" for v in (r, g, b))


def style_from_fusion(preset):
    """Map an AutoSubs preset ({'id','name','description','macroSettings'}) to a CaptionStyle."""
    m = preset.get("macroSettings") or {}
    g = lambda k, d=None: m.get(k, d)  # noqa: E731
    pos = g("TextPosition", [0.5, 0.12, 0]) or [0.5, 0.12, 0]
    style_word = str(g("Style", "Regular")).lower()
    font = str(g("Font", "Arial"))
    hl_idx = int(g("HighlightStyle", 0) or 0)
    pop, fade, slide = bool(g("PopInEnabled", 0)), bool(g("FadeEnabled", 0)), bool(g("SlideUpEnabled", 0))
    source = "autosubs" if str(preset.get("id", "")).startswith("builtin:") else "autosubs-user"
    return CaptionStyle(
        id=str(preset.get("id") or preset.get("name")), name=str(preset.get("name") or "Preset"), source=source,
        description=str(preset.get("description") or ""), font=font,
        bold="bold" in style_word or "bold" in font.lower() or "black" in font.lower(),
        size=float(g("TextSize", 0.07)), fill=_hex(g("FillColorRed", 1), g("FillColorGreen", 1), g("FillColorBlue", 1)),
        outline_on=bool(g("OutlineEnabled", 1)), outline=_hex(g("OutlineColorRed", 0), g("OutlineColorGreen", 0), g("OutlineColorBlue", 0)),
        outline_w=float(g("OutlineThickness", 0.08)), shadow_on=bool(g("ShadowEnabled", 0)),
        shadow=_hex(g("ShadowColorRed", 0), g("ShadowColorGreen", 0), g("ShadowColorBlue", 0)),
        highlight_on=bool(g("HighlightEnabled", 0)), highlight=_hex(g("HighlightColorRed", 1), g("HighlightColorGreen", 0.8), g("HighlightColorBlue", 0)),
        highlight_style=HIGHLIGHT_STYLES.get(hl_idx, "fill"), pos_x=float(pos[0]), pos_y=float(pos[1]),
        pop_in=pop, fade=fade, slide_up=slide, anim_len=float(g("AnimationLength", 0.2)), reveal="progressive" if pop else "all")


SERMON_SNIP_STYLES = [
    CaptionStyle("ss:bold_pop", "Bold pop", description="Big white capitals, the spoken word turns yellow.", font="Poppins ExtraBold", bold=False,
                 size=0.055, uppercase=True, outline_w=0.14, highlight="FFD600"),
    CaptionStyle("ss:clean", "Clean", description="Simple white captions.", font="Poppins", bold=True, size=0.042, outline_w=0.09, highlight_on=False),
    CaptionStyle("ss:boxed", "Boxed", description="White text, the spoken word on a coloured bubble.", font="Poppins", bold=True, size=0.040,
                 outline_w=0.0, outline_on=False, highlight="2563EB", highlight_style="bubble"),
    CaptionStyle("ss:karaoke", "Karaoke", description="Words light up green as they are spoken.", font="Poppins", bold=True, size=0.046,
                 outline_w=0.11, highlight="35E0A1"),
]


# ---------------------------------------------------------------- AutoSubs presets (built-in + the user's own)

def autosubs_presets_paths(extra=None):
    """Where AutoSubs keeps your presets (tauri-plugin-store: <app data>/com.autosubs/autosubs-presets.json)."""
    found = []
    if extra:
        found.append(Path(extra))
    home = Path.home()
    if sys.platform == "win32":
        roots = [Path(os.environ.get("APPDATA", str(home / "AppData" / "Roaming")))]
    elif sys.platform == "darwin":
        roots = [home / "Library" / "Application Support"]
    else:
        roots = [home / ".local" / "share", Path(os.environ.get("XDG_DATA_HOME", str(home / ".local" / "share")))]
    for root in roots:
        found.append(root / "com.autosubs" / "autosubs-presets.json")
        found += [Path(p) for p in glob.glob(str(root / "*" / "autosubs-presets.json"))]
    seen, out = set(), []
    for p in found:
        if str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def load_presets(extra_path=None, builtin_file=None):
    """All selectable styles: [{'id','name','description','source','style': CaptionStyle}], plus where user presets came from."""
    items = [{"id": s.id, "name": s.name, "description": s.description, "source": "sermonsnip", "style": s} for s in SERMON_SNIP_STYLES]
    try:
        builtin = json.loads(Path(builtin_file or BUILTIN_PRESETS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        builtin = []
    for p in builtin:
        st = style_from_fusion(p)
        items.append({"id": st.id, "name": st.name, "description": st.description, "source": "autosubs", "style": st})
    user_file, user_count = None, 0
    for path in autosubs_presets_paths(extra_path):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        presets = data.get("userPresets") if isinstance(data, dict) else data
        if not isinstance(presets, list):
            continue
        user_file = str(path)
        for p in presets:
            if isinstance(p, dict) and p.get("macroSettings") is not None:
                st = style_from_fusion(dict(p, id=p.get("id") or f"user:{p.get('name')}"))
                items.append({"id": st.id, "name": st.name, "description": st.description, "source": "autosubs-user", "style": st})
                user_count += 1
        break
    return items, {"user_file": user_file, "user_count": user_count}


def find_preset(presets, preset_id):
    for p in presets:
        if p["id"] == preset_id:
            return p["style"]
    return presets[0]["style"]


# ---------------------------------------------------------------- words -> lines

def _apply_case(word, case):
    if case == "uppercase":
        return word.upper()
    if case == "lowercase":
        return word.lower()
    if case == "titlecase":
        return word[:1].upper() + word[1:].lower()
    return word


def _display(word, fmt, style):
    text = word.replace("{", "(").replace("}", ")").replace("\\", "").strip()
    if fmt.remove_punctuation:
        text = re.sub(r"[^\w\s'’-]", "", text)
    case = fmt.text_case if fmt.text_case != "none" else ("uppercase" if style.uppercase else "none")
    return _apply_case(text, case)


def group_words(words, fmt, max_gap=0.7):
    """Blocks of words shown together. A block holds words_per_line x lines words (and obeys max_chars per line)."""
    per_line, lines = fmt.words_per_line, max(1, int(fmt.lines))
    capacity = per_line * lines
    groups, cur = [], []
    for w in words:
        if cur:
            too_long = bool(fmt.max_chars) and sum(len(x["word"]) + 1 for x in cur) + len(w["word"]) > fmt.max_chars * lines
            if len(cur) >= capacity or too_long or w["start"] - cur[-1]["end"] > max_gap or re.search(r"[.!?…]$", cur[-1]["word"]):
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


# ---------------------------------------------------------------- ASS

def _bgr(rgb):
    h = rgb.lstrip("#").upper()
    return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&"


def _bgra(rgb):
    h = rgb.lstrip("#").upper()
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}"


def _t(t):
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_ass(words, style, fmt=None, width=1080, height=1920, font=None):
    """ASS text for the words. `font` = find_font() result (decides the family name written into the file)."""
    fmt = fmt or Formatting()
    font = font or find_font(style.font, style.bold)
    size_px = max(14, int(round(height * style.size * fmt.size_scale)))
    outline_px = round(min(size_px * 0.5, max(0.0, style.outline_w * size_px)), 1) if style.outline_on else 0
    shadow_px = max(2, round(size_px * 0.06, 1)) if style.shadow_on else 0
    hl = fmt.highlight or style.highlight
    x, y = round(width * style.pos_x), round(height * (1 - style.pos_y))
    head = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {width}\nPlayResY: {height}\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
        "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font['family']},{size_px},{_bgra(style.fill)},{_bgra(hl)},{_bgra(style.outline)},{_bgra(style.shadow)},"
        f"{-1 if font['bold'] else 0},0,0,0,100,100,0,0,1,{outline_px},{shadow_px},5,{int(width * 0.05)},{int(width * 0.05)},0,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    anim_ms = max(60, int(style.anim_len * 1000))
    lines = []
    groups = group_words(words, fmt)
    for gi, group in enumerate(groups):
        limit = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else float("inf")
        texts = [_display(w["word"], fmt, style) for w in group]
        per_line = fmt.words_per_line
        n = len(group)
        for k, w in enumerate(group):
            start = w["start"]
            nxt = group[k + 1]["start"] if k + 1 < n else min(max(w["end"], start + 0.25) + 0.12, limit)
            first, last = k == 0, k == n - 1
            pos = f"\\an5\\pos({x},{y})"
            if style.slide_up and first:
                pos = f"\\an5\\move({x},{y + int(size_px * 0.6)},{x},{y},0,{anim_ms})"
            fade = ""
            if style.fade and (first or last):
                fade = f"\\fad({anim_ms if first else 0},{anim_ms if last else 0})"
            parts = []
            for j, text in enumerate(texts):
                active = style.highlight_on and j == k
                tag = ""
                if j > k and style.reveal == "progressive":
                    tag = "\\alpha&HFF&"                         # not spoken yet: invisible, but keeps the layout still
                elif active:
                    if style.highlight_style == "fill":
                        tag = f"\\c{_bgr(hl)}"
                    elif style.highlight_style == "outline":
                        tag = f"\\3c{_bgr(hl)}\\bord{round(outline_px * 1.6 + 3, 1)}"
                    elif style.highlight_style == "glow":
                        tag = f"\\3c{_bgr(hl)}\\4c{_bgr(hl)}\\blur{max(3, size_px // 12)}\\bord{round(outline_px + size_px * 0.10, 1)}"
                    else:                                         # bubble: a thick, rounded outline reads as a pill behind the word
                        tag = f"\\3c{_bgr(hl)}\\bord{round(size_px * 0.26, 1)}\\shad0"
                if j == k and style.pop_in:
                    tag += f"\\fscx60\\fscy60\\t(0,{anim_ms},\\fscx100\\fscy100)"
                parts.append(("{" + tag + "}" + text + "{\\r}") if tag else text)
                if j + 1 < n and per_line and (j + 1) % per_line == 0 and fmt.lines > 1:
                    parts.append("\\N")
                elif j + 1 < n:
                    parts.append(" ")
            body = "".join(parts).replace(" \\N ", "\\N").replace("\\N ", "\\N")
            lines.append(f"Dialogue: 0,{_t(start)},{_t(nxt)},Default,,0,0,0,,{{{pos}{fade}}}{body}")
    return head + "\n".join(lines) + "\n"


# ---------------------------------------------------------------- SRT

def _srt_time(t):
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(words, fmt=None, style=None):
    fmt = fmt or Formatting(density="more")
    style = style or SERMON_SNIP_STYLES[1]
    out = []
    for n, group in enumerate(group_words(words, fmt), 1):
        out.append(f"{n}\n{_srt_time(group[0]['start'])} --> {_srt_time(group[-1]['end'])}\n{' '.join(_display(w['word'], fmt, style) for w in group)}\n")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- encoding quality

QUALITY = {
    "maximum": {"label": "Maximum (visually lossless)", "x264": ["-preset", "slow", "-crf", "14"], "nvenc": ["-preset", "p7", "-rc", "vbr", "-cq", "15", "-b:v", "0"]},
    "high": {"label": "High (recommended)", "x264": ["-preset", "medium", "-crf", "17"], "nvenc": ["-preset", "p5", "-rc", "vbr", "-cq", "18", "-b:v", "0"]},
    "fast": {"label": "Fast (draft)", "x264": ["-preset", "veryfast", "-crf", "21"], "nvenc": ["-preset", "p3", "-rc", "vbr", "-cq", "22", "-b:v", "0"]},
}


def video_args(quality="high", nvenc=False):
    q = QUALITY.get(quality, QUALITY["high"])
    if nvenc:
        return ["-c:v", "h264_nvenc"] + q["nvenc"] + ["-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264"] + q["x264"] + ["-pix_fmt", "yuv420p"]


def nvenc_available():
    """True only if a real NVENC encode works here (the encoder can be listed yet unusable without a GPU/driver)."""
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2", "-c:v", "h264_nvenc", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=30, creationflags=_NO_WINDOW)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def burn_ass(video_path, ass_path, out_path, fonts=(), quality="high", nvenc=False, cancel=None):
    """Render video_path with the ASS captions burned in -> out_path (audio is copied, not re-encoded).

    The caption file and the given font files must live beside the video (we copy fonts there), so the filter
    arguments contain no Windows drive letters or escaping: ``ass=filename=x.ass:fontsdir=.``
    """
    from tools import ToolError, run_cancelable
    video_path, ass_path, out_path = Path(video_path), Path(ass_path), Path(out_path)
    if video_path.parent != ass_path.parent:
        raise ToolError("Internal error: video and caption file must be in the same folder.")
    for font in fonts:
        if font and Path(font).is_file():
            dest = video_path.parent / Path(font).name
            if not dest.exists():
                shutil.copyfile(font, dest)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video_path.name,
           "-vf", f"ass=filename={ass_path.name}:fontsdir=.,setsar=1"] + video_args(quality, nvenc) + \
          ["-c:a", "copy", "-movflags", "+faststart", str(out_path.resolve())]
    code, tail = run_cancelable(cmd, cancel=cancel, cwd=str(video_path.parent))
    if code != 0 or not out_path.exists():
        raise ToolError("Caption burn-in failed.\n" + "\n".join(tail[-6:]) + "\n(Run 'Test caption rendering' in the Captions step for a diagnosis.)")
    return out_path


def selftest(style=None, work=None):
    """Burn a 2-second sample and prove the captions are visible. Returns {'ok','detail','filters_ass','font'}."""
    style = style or SERMON_SNIP_STYLES[0]
    work = Path(work or tempfile.mkdtemp(prefix="ss_captest_"))
    work.mkdir(parents=True, exist_ok=True)
    info = {"ok": False, "detail": "", "filters_ass": None, "font": None, "ffmpeg": None}
    try:
        v = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True, creationflags=_NO_WINDOW)
        info["ffmpeg"] = (v.stdout.splitlines() or [""])[0]
        f = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, creationflags=_NO_WINDOW).stdout
        info["filters_ass"] = bool(re.search(r"\bass\b", f))
    except (OSError, subprocess.SubprocessError) as exc:
        info["detail"] = f"FFmpeg could not be started: {exc}"
        return info
    if not info["filters_ass"]:
        info["detail"] = "This FFmpeg build has no 'ass' filter (libass). Install the 'full' build: run INSTALL_FFMPEG.bat."
        return info
    font = find_font(style.font, style.bold)
    info["font"] = f"{font['family']} ({'found on this PC' if font['found'] else 'bundled Poppins used'})"
    base = work / "base.mp4"
    plain = work / "plain.png"
    burned_png = work / "burned.png"
    mk = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=0x203060:s=540x960:r=25:d=2", "-pix_fmt", "yuv420p", str(base)],
                        capture_output=True, text=True, creationflags=_NO_WINDOW)
    if mk.returncode != 0:
        info["detail"] = "Could not create a test video: " + mk.stderr[-300:]
        return info
    words = [{"word": w, "start": 0.1 + i * 0.4, "end": 0.4 + i * 0.4} for i, w in enumerate("Test captions are working".split())]
    ass = work / "t.ass"
    ass.write_text(build_ass(words, style, Formatting(), 540, 960, font), encoding="utf-8")
    try:
        burn_ass(base, ass, work / "out.mp4", fonts=[font["path"]], quality="fast")
    except Exception as exc:
        info["detail"] = str(exc)
        return info
    for src, dst in ((base, plain), (work / "out.mp4", burned_png)):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1.2", "-i", str(src), "-frames:v", "1", str(dst)], capture_output=True, creationflags=_NO_WINDOW)
    try:
        info["ok"] = plain.read_bytes() != burned_png.read_bytes()
    except OSError:
        info["ok"] = False
    info["detail"] = "Captions render correctly." if info["ok"] else "FFmpeg ran but no caption text was drawn (the output frame is identical to the plain video)."
    return info
