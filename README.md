# Sermon-Snip

Find the strongest moments in a sermon, cut them, reframe them to vertical with subject tracking, caption them in your
AutoSubs style, and export finished clips. DaVinci Resolve is **optional**.

```
tracks with sound ─► best moments (+ optional Gemini) ─► trim ─► reframe 9:16 (verthor) ─► captions (AutoSubs) ─► export
   auto-detected         video previews beside scores              before/after preview      your AutoSubs presets     one click: "Export all"
```

## Install (Windows)
1. Install 64-bit **Python 3.11 or 3.12** (tick *Add Python to PATH*).
2. Double-click **`INSTALL_AND_RUN.bat`**. It creates a private environment, installs everything, offers to install
   **FFmpeg** if it's missing (`INSTALL_FFMPEG.bat`, no admin needed), and opens Sermon-Snip. If it sees an NVIDIA driver it also installs the
   GPU libraries for transcription (see [GPU_SETUP.md](GPU_SETUP.md)).
3. For subject-tracking reframing run **`INSTALL_VERTHOR.bat`**. With an NVIDIA GPU it also installs the **CUDA build of PyTorch**, which is what
   makes verthor fast (see below).
4. Install **AutoSubs** (<https://github.com/tmoroney/auto-subs>) if you want AutoSubs transcription and your presets.

Everything degrades gracefully: no verthor → centre crop; no AutoSubs → Sermon-Snip's own transcript; every fallback is shown on the clip.

## The five steps (and a shortcut)
**Export all** (top right) runs every remaining stage for the selected moments in one go: reframe → captions → render. Use the steps only when you want
to look or adjust.

1. **Source.** Choose a file, or a Resolve timeline. Sermon-Snip listens to **every audio track/stream**, flags silent ones, shows a level meter, and
   pre-selects the one most likely to be the preacher (loudest, with a nudge from names like "mic"/"pastor" and away from "music"). You confirm or change it.
2. **Moments.** Ranked moments with **a video preview beside each score**. Tick the ones you want. **Adjust start/end** nudges either edge (±1 s, ±5 s) and
   rebuilds the cut from your original media. Optional **Gemini** re-ranking (see below).
3. **Reframe.** Vertical 9:16 with subject tracking, or *keep original framing*. **Before and After side by side**, played in sync.
4. **Captions** (separate from export). Your **AutoSubs presets** (built-in and your own), AutoSubs' formatting options (density, lines, words/characters per
   line, text case, remove punctuation), highlight colour and size, and a **live preview** on a real clip rendered by the same renderer the export uses.
5. **Export.** Quality (Maximum / High / Fast), NVENC if available, subtitle sidecars, folder. **Send to Resolve** is optional.

### Resolve (optional), now one script
Click **Install script**, then in Resolve run **Workspace ▸ Scripts ▸ SermonSnip** once with your timeline open: Sermon-Snip receives the timeline.
Run the same script again only when you want Sermon-Snip's moments cut into timelines or the finished clips imported. Without Resolve, choose a file instead.
The app cuts, reframes, captions and exports on its own, at full quality, from your original media.

## Your questions
**Is Gemini better for ranking?** Probably for *meaning*, not for *finding*: see [docs/RANKING.md](docs/RANKING.md). It is optional and off by default;
Sermon-Snip first shortlists ~30 candidates locally and Gemini only re-judges those (hook, completeness, impact) and writes titles. Only transcript text is sent.
On Google's free tier submitted content may be used to improve its products; the paid tier does not.

**Reframing is slow.** verthor already uses the GPU automatically, but only when PyTorch is a CUDA build, and PyPI's Windows PyTorch is CPU-only.
Sermon-Snip now detects this (Step 3 shows *"verthor runs on the CPU"*) and offers **Enable GPU**, which installs the CUDA build matching your driver
(about 2.5 GB, once). **Fast** mode is *not* a free speed-up: it lowers detection resolution to ≤512 px and turns off pose and saliency, which is why it loses the
speaker. It is off by default; with the GPU you shouldn't need it.

**Burn-in not working.** Captions now never depend on the PC's fonts: the needed font file (your preset's font if installed, otherwise the bundled Poppins) is
copied next to the subtitle file and handed to FFmpeg directly. **Test caption rendering** (Step 4) burns a sample and tells you exactly what's wrong
(missing libass, unreadable font, …). If it still fails, send me that text.

**Quality.** The cut comes from your original files. The reframed intermediate is near-lossless (CRF 14) and the final render is one more encode at your chosen
quality; with captions off the reframed file is delivered byte-for-byte (no re-encode). One thing no software can avoid: a 9:16 crop of **1080p** footage contains
only about 608×1080 pixels, which are enlarged to 1080×1920, so vertical clips are softer than the original. **4K footage** gives much sharper results. Step 5
tells you when this applies.

## Tests
`python -m unittest discover -s tests -v` (FFmpeg tests skip themselves if FFmpeg is missing). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for what is and is not verified.

## Known limitations
- Highlight scores are heuristics; review clips yourself.
- AutoSubs presets are reproduced closely, not pixel-for-pixel: Fusion's text engine only exists inside Resolve. Font, size, position, fill/outline/shadow,
  highlight Fill/Outline/Glow/Bubble and pop-in/fade/slide-up are honoured; other macro settings are ignored. A preset whose font isn't installed uses Poppins (flagged).
- Cutting into Resolve timelines skips compound/nested/generated/multicam items; retimed clips may be off.
- verthor is beta and tested upstream on macOS only; its output is a baked video (no editable keyframes).
