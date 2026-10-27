import array
import math
import sys
import wave

PHRASES = [
    "remember", "listen", "behold", "never", "always", "today", "right now",
    "i declare", "i decree", "the truth", "listen to me", "don't give up",
    "do not give up", "jesus said", "the bible says", "the word of god",
    "breakthrough", "victory", "faith", "promise", "power", "glory",
    "salvation", "holy spirit", "kingdom", "miracle", "purpose",
    "hallelujah", "amen", "somebody", "shout", "say it", "can i tell you",
    "are you hearing me"
]


def _score_rms(rms):
    """Convert per-window RMS values into normalized excitement scores."""
    if not rms:
        return []
    ordered = sorted(rms)
    baseline = max(ordered[int(.55 * (len(ordered) - 1))], .008)
    scores = [min(1.0, max(0.0, (x / baseline - .65) / 2.4)) for x in rms]
    for i in range(1, len(scores)):
        rise = max(0.0, rms[i] - rms[i - 1]) / baseline
        scores[i] = min(1.0, scores[i] + min(.22, rise * .08))
    return scores


def analyze_wav_energy(path, seconds=1, progress=None):
    """Analyze PCM WAV in bounded-memory chunks; never retain the full waveform.

    Memory use is proportional to the chunk size plus one RMS value per output
    window, rather than to the total number of audio samples.
    """
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames_total = wf.getnframes()
        if wf.getcomptype() != "NONE":
            raise ValueError("Please use uncompressed PCM WAV audio.")
        if width != 2:
            raise ValueError(
                f"Expected 16-bit PCM WAV, but this file uses {width * 8}-bit samples. "
                "Export/convert an analysis copy to 16-bit PCM WAV."
            )
        if channels < 1 or rate < 1:
            raise ValueError("The WAV file has invalid channel or sample-rate metadata.")

        window_frames = max(1, int(rate * seconds))
        chunk_frames = max(1, rate * 10)
        rms = []
        sum_squares = 0.0
        frames_in_window = 0
        frames_read = 0

        while True:
            raw = wf.readframes(chunk_frames)
            if not raw:
                break
            pcm = array.array("h")
            pcm.frombytes(raw)
            if sys.byteorder != "little":
                pcm.byteswap()
            frame_count = len(pcm) // channels
            if frame_count == 0:
                continue

            # Process one audio frame at a time; do not create per-frame slices.
            for frame in range(frame_count):
                base = frame * channels
                total = 0
                for channel in range(channels):
                    total += pcm[base + channel]
                mono = total / (channels * 32768.0)
                sum_squares += mono * mono
                frames_in_window += 1
                frames_read += 1
                if frames_in_window >= window_frames:
                    rms.append(math.sqrt(sum_squares / frames_in_window))
                    sum_squares = 0.0
                    frames_in_window = 0

            if progress and frames_total:
                progress(min(1.0, frames_read / frames_total))

        if frames_in_window:
            rms.append(math.sqrt(sum_squares / frames_in_window))

        duration = frames_total / rate
        return duration, _score_rms(rms)


def read_pcm_wav(path):
    """Compatibility helper: load mono WAV into a compact array.

    The app's main analysis path uses analyze_wav_energy() instead, which is
    bounded-memory and should be preferred for long recordings.
    """
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        if wf.getcomptype() != "NONE":
            raise ValueError("Please use uncompressed PCM WAV audio.")
        if width != 2:
            raise ValueError(
                f"Expected 16-bit PCM WAV, but this file uses {width * 8}-bit samples. "
                "Export/convert an analysis copy to 16-bit PCM WAV."
            )
        samples = array.array("f")
        chunk_frames = max(1, rate * 10)
        while True:
            raw = wf.readframes(chunk_frames)
            if not raw:
                break
            pcm = array.array("h")
            pcm.frombytes(raw)
            if sys.byteorder != "little":
                pcm.byteswap()
            if channels == 1:
                samples.extend(v / 32768.0 for v in pcm)
            else:
                for i in range(0, len(pcm) - channels + 1, channels):
                    total = 0
                    for channel in range(channels):
                        total += pcm[i + channel]
                    samples.append(total / (channels * 32768.0))
        return rate, samples


def energy_windows(samples, rate, seconds=1):
    """Compute RMS windows for an already-loaded mono sample array."""
    size = max(1, int(rate * seconds))
    rms = []
    for start in range(0, len(samples), size):
        stop = min(start + size, len(samples))
        total = 0.0
        for i in range(start, stop):
            value = samples[i]
            total += value * value
        count = stop - start
        if count:
            rms.append(math.sqrt(total / count))
    return _score_rms(rms)


def _prepare_windows_cuda_dll_search():
    """Make cuBLAS/cuDNN loadable by CTranslate2 (PATH + CUDA_PATH + add_dll_directory + preload).

    The old version only called os.add_dll_directory, which CTranslate2's own runtime loader
    does not consult. See gpu_setup.py for the details. Returns the list of folders applied.
    """
    import gpu_setup
    return gpu_setup.prepare_cuda_environment().path_dirs


WHISPER_RATE = 16000


def _to_16k(audio, rate, np):
    """Resample mono float32 audio to Whisper's 16 kHz. Whisper treats raw arrays as 16 kHz, so
    feeding 44.1/48 kHz samples unconverted plays the speech back at the wrong speed."""
    if rate == WHISPER_RATE:
        return audio
    if rate % WHISPER_RATE == 0:                     # 48 kHz -> 16 kHz: average each group of 3 (box filter)
        k = rate // WHISPER_RATE
        usable = len(audio) // k * k
        return audio[:usable].reshape(-1, k).mean(axis=1).astype(np.float32)
    n_out = int(len(audio) * WHISPER_RATE / rate)    # 44.1 kHz etc.: linear interpolation
    if n_out <= 0:
        return audio[:0]
    positions = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


class GpuError(RuntimeError):
    """NVIDIA GPU transcription could not start or failed. `.diagnosis` is a readable report."""

    def __init__(self, message, diagnosis=""):
        super().__init__(message)
        self.diagnosis = diagnosis


def _gpu_failure(message, original):
    import gpu_setup
    try:
        diagnosis = gpu_setup.diagnose(prepare=False).text()
    except Exception as exc:  # diagnosis must never hide the real error
        diagnosis = f"(diagnosis unavailable: {exc})"
    return GpuError(f"{message}\n\nOriginal error: {original}", diagnosis)


def transcribe(path, model_name="large-v3", progress=None, language="en",
              device="auto", chunk_seconds=30, overlap_seconds=2,
              initial_prompt=None):
    """Transcribe PCM WAV in overlapping chunks with conservative hallucination controls.

    GPU mode is strict when device='cuda': it never silently falls back to CPU.
    Each overlap is assigned to one chunk using segment midpoint ownership,
    preventing duplicate transcript segments while preserving chunk context.
    """
    if device in ("auto", "cuda"):
        _prepare_windows_cuda_dll_search()
    try:
        import numpy as np
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper and NumPy are required. Run INSTALL_AND_RUN.bat."
        ) from exc

    with wave.open(str(path), "rb") as wf:
        if wf.getcomptype() != "NONE" or wf.getsampwidth() != 2:
            raise ValueError("Transcription requires uncompressed 16-bit PCM WAV audio.")
        channels = wf.getnchannels()
        rate = wf.getframerate()
        total_frames = wf.getnframes()
        if channels < 1 or rate < 1:
            raise ValueError("The WAV file has invalid channel or sample-rate metadata.")

    chunk_frames = max(1, rate * int(chunk_seconds))
    overlap_frames = max(0, rate * int(overlap_seconds))
    step_frames = max(1, chunk_frames - overlap_frames)

    if progress:
        progress(f"Loading Whisper model {model_name}…")

    errors = []
    attempts = [("cuda", "float16"), ("cuda", "int8_float16")] if device in ("auto", "cuda") else [("cpu", "int8")]
    model = None
    chosen_device = "cuda" if device in ("auto", "cuda") else "cpu"
    for target_device, compute_type in attempts:
        try:
            model = WhisperModel(model_name, device=target_device, compute_type=compute_type)
            chosen_device = target_device
            if progress:
                progress(f"Whisper ready: {model_name} on {target_device} ({compute_type})")
            break
        except Exception as exc:
            errors.append(f"{target_device}/{compute_type}: {exc}")
            model = None
    if model is None:
        detail = " | ".join(errors[-2:])
        if chosen_device == "cuda":
            raise _gpu_failure(
                "NVIDIA GPU transcription could not start, and no CPU fallback was used "
                "(choose Device: CPU to run on the processor instead).", detail)
        raise RuntimeError("Could not initialize Whisper: " + detail)

    rows = []
    detected_language = language or "unknown"
    start_frame = 0
    chunk_index = 0

    with wave.open(str(path), "rb") as wf:
        while start_frame < total_frames:
            wf.setpos(start_frame)
            raw = wf.readframes(min(chunk_frames, total_frames - start_frame))
            if not raw:
                break
            pcm = np.frombuffer(raw, dtype="<i2")
            frame_count = len(pcm) // channels
            if frame_count == 0:
                break
            mono = pcm[:frame_count * channels].reshape(frame_count, channels).astype(np.float32).mean(axis=1)
            audio = _to_16k(mono / 32768.0, rate, np)
            chunk_start = start_frame / rate
            chunk_end = (start_frame + frame_count) / rate
            is_last = start_frame + frame_count >= total_frames

            options = {
                "language": language or None,
                "beam_size": 5,
                "temperature": 0.0,
                "condition_on_previous_text": False,
                "vad_filter": True,
                "word_timestamps": False,
            }
            if initial_prompt:
                options["initial_prompt"] = initial_prompt

            try:
                segments, info = model.transcribe(audio, **options)
                # faster-whisper yields segments lazily; CUDA library failures can
                # therefore occur here, not during model construction.
                segments = list(segments)
            except RuntimeError as exc:
                message = str(exc).lower()
                cuda_failure = chosen_device == "cuda" and (
                    "cublas" in message or "cudnn" in message
                    or "cuda" in message or "cannot be loaded" in message
                )
                if not cuda_failure:
                    raise
                raise _gpu_failure(
                    "NVIDIA GPU transcription failed while processing audio. Sermon-Snip did not switch to CPU.",
                    exc) from exc

            if detected_language in ("unknown", None, ""):
                detected_language = getattr(info, "language", "unknown")

            # The overlap belongs to the earlier chunk up to the next chunk's
            # start plus overlap. Keep a segment only in the chunk containing its midpoint.
            ownership_end = total_frames / rate if is_last else (start_frame + step_frames) / rate
            for seg in segments:
                local_start = float(seg.start)
                local_end = float(seg.end)
                midpoint = chunk_start + (local_start + local_end) / 2
                if midpoint < chunk_start - 0.01 or midpoint >= ownership_end:
                    continue
                global_start = chunk_start + local_start
                global_end = min(chunk_end, chunk_start + local_end)
                text = seg.text.strip()
                if text and global_end > global_start:
                    rows.append({
                        "start": round(max(0.0, global_start), 3),
                        "end": round(global_end, 3),
                        "text": text,
                    })

            chunk_index += 1
            if progress and total_frames:
                progress(
                    f"Transcribing ({chosen_device})… "
                    f"{min(1.0, (start_frame + frame_count) / total_frames):.0%}"
                )
            if is_last:
                break
            start_frame += step_frames

    rows.sort(key=lambda item: (item["start"], item["end"]))
    # Filter exact adjacent duplicate text sometimes emitted by chunk-edge decoding.
    deduped = []
    for row in rows:
        if deduped and row["text"].casefold() == deduped[-1]["text"].casefold() and row["start"] < deduped[-1]["end"] + 0.35:
            if row["end"] > deduped[-1]["end"]:
                deduped[-1]["end"] = row["end"]
            continue
        deduped.append(row)

    if not deduped:
        raise RuntimeError("No speech detected in this recording.")
    return deduped, detected_language


def text_score(text):
    low = text.lower()
    hits = [phrase for phrase in PHRASES if phrase in low]
    score = min(1.0, len(hits) * .13 + (.12 if "!" in text else 0) + (.06 if "?" in text else 0))
    if any(low.startswith(prefix) for prefix in ("but ", "because ", "when ", "if ", "you ", "god ", "jesus ")):
        score = min(1.0, score + .08)
    return score, hits[:5]


def _ends_sentence(text):
    """Whether transcript text appears to end a complete sentence/thought."""
    return bool(text.rstrip().endswith((".", "!", "?", "…", '"', "”")))


def candidates(transcript, energy, duration, limit=30, target_seconds=60):
    """Find impactful excerpts near a minute, preferring complete sentence boundaries.

    The transcript model does not provide true semantic discourse boundaries, so
    punctuation and pauses are used as practical boundary signals. Shorter clips
    remain eligible when they score strongly.
    """
    if not transcript:
        return []

    # Start candidate windows at sentence boundaries, while tolerating ASR segments
    # that omit punctuation. A long pause is also a reasonable boundary.
    starts = []
    for i, item in enumerate(transcript):
        if i == 0:
            starts.append(i)
            continue
        previous = transcript[i - 1]
        gap = item["start"] - previous["end"]
        if _ends_sentence(previous["text"]) or gap >= 1.0:
            starts.append(i)

    raw_results = []
    seen_ranges = set()
    min_seconds = 12.0
    max_seconds = 82.0

    for first in starts:
        start_time = max(0.0, float(transcript[first]["start"]) - 0.5)
        best = None
        for last in range(first, len(transcript)):
            item = transcript[last]
            if last > first and item["start"] - transcript[last - 1]["end"] >= 2.8:
                break
            end_time = min(duration, float(item["end"]) + 0.5)
            length = end_time - start_time
            if length > max_seconds:
                break
            if length < min_seconds:
                continue

            # Prefer thought-ending punctuation. Near the target, allow the
            # nearest segment boundary if Whisper omitted punctuation.
            sentence_end = _ends_sentence(item["text"])
            if not sentence_end and abs(length - target_seconds) > 8:
                continue

            text = " ".join(x["text"].strip() for x in transcript[first:last + 1]).strip()
            if len(text.split()) < 12:
                continue
            # Penalize windows that are far from the target, but allow genuinely
            # short, punchy excerpts to remain competitive.
            duration_fit = max(0.45, 1.0 - abs(length - target_seconds) / target_seconds)
            lo = max(0, int(start_time))
            hi = min(len(energy), max(lo + 1, math.ceil(end_time)))
            vals = energy[lo:hi]
            audio = sum(vals) / len(vals) if vals else 0.0
            peak = max(vals) if vals else 0.0
            spoken, hits = text_score(text)
            completeness = 1.0 if sentence_end else 0.78
            score = 100 * (.36 * audio + .42 * spoken + .14 * peak + .08 * duration_fit)
            score *= completeness
            key = (round(start_time), round(end_time))
            if key in seen_ranges:
                continue
            result = {
                "start": round(start_time, 2), "end": round(end_time, 2),
                "duration": round(length, 1), "score": round(score, 1),
                "audio_score": round(audio, 3), "text_score": round(spoken, 3),
                "peak_score": round(peak, 3), "text": text, "signals": hits,
                "complete_thought": sentence_end,
            }
            # Keep the strongest candidate for each start boundary, favoring
            # a complete ending and proximity to about one minute.
            rank = score + (5 if sentence_end else 0) - abs(length - target_seconds) * 0.04
            if best is None or rank > best[0]:
                best = (rank, result, key)
            # Once we're well beyond target, don't keep expanding this excerpt.
            if length >= target_seconds + 10 and sentence_end:
                break
        if best:
            seen_ranges.add(best[2])
            raw_results.append(best[1])

    # Rank candidates by score (highest first), while reducing near-duplicates.
    raw_results.sort(key=lambda item: item["score"], reverse=True)
    chosen = []
    for item in raw_results:
        # Keep only non-overlapping clips. (The previous version also accepted near-duplicates
        # because the 'near-duplicate' clause sat inside all(...) and so let them THROUGH.)
        if all(item["end"] <= old["start"] or item["start"] >= old["end"] for old in chosen):
            chosen.append(item)
        if len(chosen) >= limit:
            break
    return chosen
