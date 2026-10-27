"""Word handling for captions (the rendering lives in caption_style.py)."""
from caption_style import (CaptionStyle, Formatting, SERMON_SNIP_STYLES, build_ass, build_srt, burn_ass, find_font,  # noqa: F401
                           find_preset, group_words, load_presets, selftest, style_from_fusion)

__all__ = ["CaptionStyle", "Formatting", "SERMON_SNIP_STYLES", "build_ass", "build_srt", "burn_ass", "find_font", "find_preset", "group_words",
           "load_presets", "selftest", "style_from_fusion", "words_from_autosubs", "words_from_segments", "spread_words"]


# ---------------------------------------------------------------- words

def words_from_autosubs(data):
    """Flat list of {'word','start','end'} from an AutoSubs transcript JSON (prefers raw segments)."""
    words = []
    for seg in (data.get("originalSegments") or data.get("segments") or []):
        if seg.get("words"):
            words += [{"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])}
                      for w in seg["words"] if w["word"].strip()]
        else:
            words += spread_words(seg["text"], float(seg["start"]), float(seg["end"]))
    return words


def spread_words(text, start, end):
    """No word timings available: spread the words across the segment in proportion to their length."""
    toks = text.split()
    if not toks:
        return []
    total = sum(len(t) + 1 for t in toks)
    out, cursor = [], start
    for t in toks:
        d = (end - start) * (len(t) + 1) / total
        out.append({"word": t, "start": cursor, "end": cursor + d})
        cursor += d
    return out


def words_from_segments(segments, start=0.0, end=None):
    """Words for the clip [start, end] of a plain transcript, shifted so the clip begins at 0."""
    end = float("inf") if end is None else end
    out = []
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        for w in spread_words(seg["text"], seg["start"], seg["end"]):
            if w["end"] > start and w["start"] < end:
                out.append({"word": w["word"], "start": max(0.0, w["start"] - start), "end": min(end, w["end"]) - start})
    return out
