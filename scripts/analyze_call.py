"""Analyze a voice-evolve duet call by VAD-slicing each channel of the
per-side stereo wavs and running Qwen3-ASR on each slice.

Inputs: --run-id YYYYMMDDTHHMMSSZ  (matches duet_<run_id>_{katey,chippy}_stereo.wav)

Outputs (under recordings/):
    duet_<run_id>_analysis.txt        — chronological merged timeline
    duet_<run_id>_analysis.jsonl      — raw timestamped utterances per source
    duet_<run_id>_metrics.json        — turn-taking latency, overlap %, e2e delay

Method:
- Each side's stereo wav has channel L = its local TTS, channel R = peer audio.
- "Katey local" (katey_L) and "Chippy peer" (chippy_R) should be the same
  signal seen on either end of the call, offset by the Telegram one-way delay.
- VAD-slice each channel into speech segments → ASR each slice → timestamped
  utterance.
- Merge into one timeline; compute:
    * turn-taking latency: time from peer utterance END → local response START
    * overlap %: fraction of wall-clock where both katey_L and chippy_L are active
    * e2e delay: median offset between katey_L and chippy_R utterance starts

Usage:
    uv run --project lintvoiceagent python scripts/analyze_call.py \\
        --run-id 20260522T162220Z
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import wave
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lintvoiceagent"))
REC_DIR = REPO_ROOT / "recordings"

CALL_RATE = 48000
VAD_RATE = 16000


@dataclass
class Utterance:
    source: str            # "katey_local"|"katey_peer"|"chippy_local"|"chippy_peer"
    start_s: float
    end_s: float
    text: str

    def as_json(self): return asdict(self)


# ---------- I/O ----------

def load_stereo_wav(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 2, f"{path} is not stereo"
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
    left = a[:, 0].astype(np.float32) / 32768.0   # local TTS
    right = a[:, 1].astype(np.float32) / 32768.0  # peer
    return left, right, sr


def resample_to(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32)
    # cheap linear resample via numpy
    duration = audio.shape[0] / sr_in
    n_out = int(round(duration * sr_out))
    t_out = np.linspace(0, duration, num=n_out, endpoint=False)
    t_in = np.linspace(0, duration, num=audio.shape[0], endpoint=False)
    return np.interp(t_out, t_in, audio).astype(np.float32)


# ---------- VAD slicing ----------

def vad_segments(audio_16k: np.ndarray, vad,
                 win_s: float = 0.25,
                 min_seg_s: float = 0.6,
                 silence_join_s: float = 0.4,
                 energy_floor: float = 0.005) -> list[tuple[float, float]]:
    """Slide a window over audio_16k. Return [(start_s, end_s), …]."""
    win = int(win_s * VAD_RATE)
    hop = win  # non-overlapping windows
    n = audio_16k.shape[0]
    flags: list[bool] = []
    for i in range(0, n - win + 1, hop):
        w = audio_16k[i:i + win]
        rms = float(np.sqrt(np.mean(w * w)))
        if rms < energy_floor:
            flags.append(False)
        else:
            flags.append(bool(vad.has_speech(w)))

    # Merge consecutive Trues, allowing small silence gaps.
    segs: list[tuple[float, float]] = []
    silence_join_wins = max(1, int(silence_join_s / win_s))
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        # absorb internal silences shorter than silence_join_s
        while j < len(flags):
            if flags[j]:
                j += 1
                continue
            # peek ahead
            k = j
            while k < len(flags) and not flags[k]:
                k += 1
            if k - j <= silence_join_wins and k < len(flags):
                j = k  # join through the gap
            else:
                break
        start_s = i * win_s
        end_s = j * win_s
        if end_s - start_s >= min_seg_s:
            segs.append((start_s, end_s))
        i = j + 1
    return segs


# ---------- ASR per slice ----------

def asr_slice(asr_model, asr_generate, audio_16k: np.ndarray,
              start_s: float, end_s: float) -> str:
    a = audio_16k[int(start_s * VAD_RATE):int(end_s * VAD_RATE)]
    if a.size == 0:
        return ""
    pcm16 = np.clip(a * 32767.0, -32768, 32767).astype(np.int16).tobytes()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(VAD_RATE)
            wf.writeframes(pcm16)
        r = asr_generate(model=asr_model, audio=tmp.name, format="txt",
                         verbose=False)
        return (getattr(r, "text", "") or "").strip()
    finally:
        try: os.unlink(tmp.name)
        except Exception: pass


# ---------- metrics ----------

def compute_overlap(local_a: list[Utterance], local_b: list[Utterance]) -> float:
    """Fraction of wall-clock time where BOTH parties have a local-speaking
    utterance active, measured over the union of speaking windows."""
    if not local_a or not local_b:
        return 0.0
    # Build event list (start, +1, who), (end, -1, who)
    events = []
    for u in local_a: events.append((u.start_s, 'a', +1)); events.append((u.end_s, 'a', -1))
    for u in local_b: events.append((u.start_s, 'b', +1)); events.append((u.end_s, 'b', -1))
    events.sort()
    a_active = 0; b_active = 0
    last_t = events[0][0]
    overlap_s = 0.0
    union_s = 0.0
    for t, who, delta in events:
        dt = t - last_t
        if dt > 0:
            if a_active > 0 and b_active > 0:
                overlap_s += dt
            if a_active > 0 or b_active > 0:
                union_s += dt
        if who == 'a': a_active += delta
        else: b_active += delta
        last_t = t
    return (overlap_s / union_s) if union_s > 0 else 0.0


def turn_latencies(local_self: list[Utterance],
                   peer_heard: list[Utterance]) -> list[float]:
    """For each self utterance, find the most recent peer-heard utterance
    that ended BEFORE self.start; latency = self.start - peer.end."""
    out = []
    for s in local_self:
        candidates = [p for p in peer_heard if p.end_s <= s.start_s]
        if not candidates:
            continue
        nearest = max(candidates, key=lambda p: p.end_s)
        lat = s.start_s - nearest.end_s
        if 0 <= lat <= 15:  # plausible window
            out.append(lat)
    return out


def e2e_delays(local_speaker: list[Utterance],
               peer_listener: list[Utterance],
               max_drift_s: float = 8.0) -> list[float]:
    """Pair each `local_speaker` utterance with the nearest `peer_listener`
    utterance whose start is later but within max_drift_s. Returns offsets
    (peer.start - local.start), i.e. one-way transmission delay."""
    out = []
    for u in local_speaker:
        cands = [p for p in peer_listener
                 if 0 <= p.start_s - u.start_s <= max_drift_s]
        if not cands:
            continue
        nearest = min(cands, key=lambda p: abs(p.start_s - u.start_s))
        out.append(nearest.start_s - u.start_s)
    return out


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    katey_wav = REC_DIR / f"duet_{args.run_id}_katey_stereo.wav"
    chippy_wav = REC_DIR / f"duet_{args.run_id}_chippy_stereo.wav"
    if not (katey_wav.exists() and chippy_wav.exists()):
        print(f"missing stereo wavs:\n  {katey_wav}\n  {chippy_wav}")
        print("Note: the duet must have been recorded with the stereo "
              "recording change in run_agent.py.")
        return

    print(f"loading wavs:\n  {katey_wav}\n  {chippy_wav}")
    kL_48, kR_48, sr = load_stereo_wav(katey_wav)
    cL_48, cR_48, _ = load_stereo_wav(chippy_wav)
    print(f"katey: {kL_48.shape[0] / sr:.1f}s   chippy: {cL_48.shape[0] / sr:.1f}s")

    # Downsample each channel to 16k for VAD/ASR.
    kL = resample_to(kL_48, sr, VAD_RATE)
    kR = resample_to(kR_48, sr, VAD_RATE)
    cL = resample_to(cL_48, sr, VAD_RATE)
    cR = resample_to(cR_48, sr, VAD_RATE)

    print("loading FireRedVAD + Qwen3-ASR…")
    from vad_detector import FireRedVAD
    from mlx_audio.stt.utils import load_model as load_asr_model
    from mlx_audio.stt.generate import generate_transcription
    vad = FireRedVAD(speech_threshold=0.4, silence_threshold=4, min_speech_frame=6)
    asr = load_asr_model("mlx-community/Qwen3-ASR-0.6B-4bit")

    def slice_and_asr(audio: np.ndarray, source: str) -> list[Utterance]:
        segs = vad_segments(audio, vad)
        utts: list[Utterance] = []
        for s, e in segs:
            txt = asr_slice(asr, generate_transcription, audio, s, e)
            if not txt or txt.lower() in {"the.", "thank you.", "you.", "..."}:
                continue
            utts.append(Utterance(source, s, e, txt))
            print(f"  [{source:13s}] {s:6.2f}-{e:6.2f}  {txt!r}")
        vad.reset()
        return utts

    print("\n--- katey local (her TTS) ---")
    k_local = slice_and_asr(kL, "katey_local")
    print("--- katey peer (what katey heard) ---")
    k_peer = slice_and_asr(kR, "katey_peer")
    print("--- chippy local (her TTS) ---")
    c_local = slice_and_asr(cL, "chippy_local")
    print("--- chippy peer (what chippy heard) ---")
    c_peer = slice_and_asr(cR, "chippy_peer")

    all_utts = k_local + k_peer + c_local + c_peer
    all_utts.sort(key=lambda u: u.start_s)

    # --- metrics ---
    overlap = compute_overlap(k_local, c_local)
    katey_lat = turn_latencies(k_local, k_peer)   # katey's reply latencies
    chippy_lat = turn_latencies(c_local, c_peer)  # chippy's reply latencies
    # e2e delay: katey says X locally; chippy hears X (peer). Their wall clocks
    # are not aligned, so this measures the offset in each side's local timeline.
    # (Useful as a relative signal; absolute e2e needs a shared clock.)
    e2e_k_to_c = e2e_delays(k_local, c_peer)
    e2e_c_to_k = e2e_delays(c_local, k_peer)

    def stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        arr = np.array(xs)
        return {"n": len(xs), "mean_s": float(arr.mean()),
                "median_s": float(np.median(arr)),
                "p90_s": float(np.percentile(arr, 90)),
                "min_s": float(arr.min()), "max_s": float(arr.max())}

    metrics = {
        "run_id": args.run_id,
        "talk_overlap_fraction": overlap,
        "katey_response_latency": stats(katey_lat),
        "chippy_response_latency": stats(chippy_lat),
        "e2e_offset_katey_to_chippy": stats(e2e_k_to_c),
        "e2e_offset_chippy_to_katey": stats(e2e_c_to_k),
        "n_utterances": {
            "katey_local": len(k_local), "katey_peer": len(k_peer),
            "chippy_local": len(c_local), "chippy_peer": len(c_peer),
        },
    }
    metrics_path = REC_DIR / f"duet_{args.run_id}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nmetrics -> {metrics_path}")
    print(json.dumps(metrics, indent=2))

    # --- raw jsonl ---
    jsonl_path = REC_DIR / f"duet_{args.run_id}_analysis.jsonl"
    with open(jsonl_path, "w") as f:
        for u in all_utts:
            f.write(json.dumps(u.as_json()) + "\n")
    print(f"raw utterances -> {jsonl_path}")

    # --- chronological merged timeline ---
    timeline_path = REC_DIR / f"duet_{args.run_id}_analysis.txt"
    with open(timeline_path, "w") as f:
        f.write(f"# duet {args.run_id} — analyzed timeline (per-channel)\n")
        f.write(f"# columns: t_start  duration  source            text\n\n")
        for u in all_utts:
            f.write(f"{u.start_s:7.2f}  {u.end_s - u.start_s:5.2f}  "
                    f"{u.source:13s}  {u.text}\n")
    print(f"timeline   -> {timeline_path}")


if __name__ == "__main__":
    main()
