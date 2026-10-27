"""Optional: let Google's Gemini re-judge the shortlisted moments.

Why a re-ranker and not a replacement: Sermon-Snip's own scoring (delivery energy + key phrases + sentence
boundaries) is free, offline and good at *finding* candidate stretches. It cannot judge meaning: whether a stretch is a
complete thought, has a hook, would make sense to a stranger scrolling past. A language model can. So the local scorer
shortlists (about 30 candidates) and Gemini scores, titles and explains them; the two scores are blended.

Privacy: this sends the shortlisted transcript text (not audio or video) to Google. On the Gemini API's free tier Google
may use submitted content to improve its products; the paid tier does not. The feature is off unless you add a key.
"""
import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = "gemini-flash-latest"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"
MAX_WORDS = 170            # per candidate; a 60 s clip is ~150 words

PROMPT = """You are an editor choosing the best short vertical-video clips (TikTok / Reels / Shorts) from a church sermon.

Below are candidate clips, each with an id and its transcript. Judge each one as a STAND-ALONE clip for someone who has never
seen the sermon. Consider: a strong hook in the first sentence; a complete thought with a clear beginning and end; one clear
message; emotional or spiritual impact; quotable lines; no dependence on earlier context ("as I said before", "turn to page");
no housekeeping (announcements, offering, greetings).

Return JSON: an array with one object per candidate:
  id (integer, copied exactly), score (0-100, be discriminating: use the full range), title (max 8 words, a compelling hook, no
  hashtags), reason (one short sentence).

Candidates:
"""

SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "id": {"type": "INTEGER"}, "score": {"type": "NUMBER"}, "title": {"type": "STRING"}, "reason": {"type": "STRING"}},
    "required": ["id", "score", "title", "reason"]}}


class GeminiError(RuntimeError):
    pass


def get_key(config=None):
    return (os.environ.get("GEMINI_API_KEY") or (config or {}).get("gemini_key") or "").strip()


def _call(url, body=None, key="", timeout=90):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        try:
            detail = json.loads(detail)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
        if exc.code in (400, 403) and "api key" in detail.lower():
            raise GeminiError("Gemini rejected the API key. Check it in Google AI Studio.") from exc
        if exc.code == 429:
            raise GeminiError("Gemini says the rate limit or free quota is used up. Wait a minute or switch to a paid key.") from exc
        raise GeminiError(f"Gemini error {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GeminiError(f"Could not reach Gemini: {exc}") from exc


def test_key(key, endpoint=ENDPOINT):
    """Cheap check that the key works: list models (no tokens spent). Returns a few model names."""
    if not key:
        raise GeminiError("Enter a Gemini API key first.")
    data = _call(f"{endpoint}/models?pageSize=50", None, key, timeout=20)
    names = [m.get("name", "").replace("models/", "") for m in data.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
    return names


def _clip_text(text):
    words = text.split()
    return " ".join(words[:MAX_WORDS]) + (" …" if len(words) > MAX_WORDS else "")


def rerank(moments, key, model=DEFAULT_MODEL, endpoint=ENDPOINT, timeout=120):
    """Ask Gemini to score/title the moments. Returns {id: {'score','title','reason'}} (only ids that were sent)."""
    if not key:
        raise GeminiError("No Gemini API key set.")
    if not moments:
        return {}
    listing = "\n\n".join(f"[id {m['id']}] ({round(m['end'] - m['start'])}s)\n{_clip_text(m['text'])}" for m in moments)
    body = {"contents": [{"role": "user", "parts": [{"text": PROMPT + listing}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json", "responseSchema": SCHEMA}}
    data = _call(f"{endpoint}/models/{model}:generateContent", body, key, timeout)
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        rows = json.loads(text)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        reason = (data.get("promptFeedback") or {}).get("blockReason")
        raise GeminiError("Gemini returned no usable ranking" + (f" (blocked: {reason})" if reason else "") + ".") from exc
    known = {m["id"] for m in moments}
    out = {}
    for row in rows if isinstance(rows, list) else []:
        try:
            mid = int(row["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if mid in known and mid not in out:
            out[mid] = {"score": max(0.0, min(100.0, float(row.get("score", 0)))), "title": str(row.get("title", "")).strip()[:90],
                        "reason": str(row.get("reason", "")).strip()[:220]}
    if not out:
        raise GeminiError("Gemini's answer did not mention any of the candidate ids.")
    return out


def blend(local_score, ai_score, weight=0.7):
    """Final score: weight on Gemini's judgement, the rest on the local score."""
    return round(weight * ai_score + (1 - weight) * local_score, 1)
