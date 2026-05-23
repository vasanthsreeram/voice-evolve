"""Use Gemini as a judge on a voice-evolve duet call recording.

Takes a run_id, downmixes one side's stereo wav (L=local TTS, R=peer audio)
into a mono mix containing the full conversation, uploads it to Gemini, and
asks for a structured evaluation: pacing, overlap, coherence, observed latency,
and concrete recommendations.

Outputs:
    recordings/duet_<run_id>_judgment.json     structured judgment from Gemini
    stdout: pretty-printed summary

Usage:
    uv run --project lintvoiceagent python scripts/judge_call.py --run-id <id>
        [--model gemini-3-flash-preview]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
REC_DIR = REPO_ROOT / "recordings"
load_dotenv(REPO_ROOT / ".env")


def stereo_to_mono_mix(stereo_wav: Path, out_wav: Path):
    """Downmix L+R of a stereo wav into a single mono wav (sum then clip)."""
    with wave.open(str(stereo_wav), "rb") as wf:
        assert wf.getnchannels() == 2, f"{stereo_wav} not stereo"
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    a = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).astype(np.int32)
    mono = (a[:, 0] + a[:, 1])
    mono = np.clip(mono, -32768, 32767).astype(np.int16)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(mono.tobytes())


PROMPT = """\
You are evaluating a recorded phone call between two AI agents, "Katey" (pro-choice)
and "Chippy" (pro-life), who are debating abortion. Both sides run a local
ASR → LLM → TTS pipeline over a real Telegram MTProto call. Your job is to
listen carefully and describe what you hear — the optimization decisions are made
by another system that reads your feedback.

Return JSON with this schema (only these fields, no others):

{
  "passed": boolean,                     // does this sound like a natural, well-paced human debate?
  "passed_reasoning": string,            // one sentence explaining the pass/fail
  "pacing_score": integer (0-10),        // 10 = natural turn-taking, no awkward gaps or rushes
  "overlap_score": integer (0-10),       // 10 = no talk-over; both wait their turn
  "coherence_score": integer (0-10),     // 10 = they actually engage with each other's specific points
  "naturalness_score": integer (0-10),   // 10 = sounds human, prosody/intonation/word choice
  "observed_response_latency_s": number, // typical gap (seconds) between one speaker finishing and the other starting
  "issues": [string, ...]                // descriptive observations of what's wrong. Be specific —
                                         // e.g. "long ~3 second silences between turns", "Katey
                                         // interrupts Chippy at 0:42", "Chippy's response at 1:10
                                         // ignores Katey's previous question about ER visits",
                                         // "Kokoro voice is robotic during long phrases".
                                         // Do NOT prescribe fixes. Just describe what you observed.
}

`passed = true` ONLY IF: pacing_score >= 8 AND overlap_score >= 8 AND
coherence_score >= 7 AND naturalness_score >= 6 AND observed_response_latency_s <= 2.5.

Reply with JSON only, no commentary.
"""


def upload_and_judge(audio_path: Path, model: str):
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set (check .env)")
    client = genai.Client(api_key=api_key)

    print(f"[judge] uploading {audio_path} ({audio_path.stat().st_size:,} bytes)…")
    myfile = client.files.upload(file=str(audio_path))
    print(f"[judge] uploaded → {myfile.uri}")

    print(f"[judge] asking {model} to evaluate…")
    resp = client.models.generate_content(
        model=model,
        contents=[PROMPT, myfile],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    return resp.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--side", default="katey", choices=("katey", "chippy"),
                    help="which side's stereo wav to feed Gemini")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    args = ap.parse_args()

    stereo = REC_DIR / f"duet_{args.run_id}_{args.side}_stereo.wav"
    if not stereo.exists():
        raise SystemExit(f"missing {stereo}")

    # Downmix to mono so the file is small and both speakers are audible.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        mono_path = Path(tmp.name)
    try:
        stereo_to_mono_mix(stereo, mono_path)

        # Try the user-requested model first; if it 404s, fall back.
        candidates = [args.model, "gemini-3.5-flash", "gemini-2.5-flash"]
        seen = set()
        candidates = [m for m in candidates if not (m in seen or seen.add(m))]
        last_err = None
        for m in candidates:
            try:
                text = upload_and_judge(mono_path, m)
                judgment = json.loads(text)
                judgment["_model_used"] = m
                judgment["_run_id"] = args.run_id
                out_path = REC_DIR / f"duet_{args.run_id}_judgment.json"
                with open(out_path, "w") as f:
                    json.dump(judgment, f, indent=2)
                print(f"\n=== JUDGMENT (model={m}) ===")
                print(json.dumps(judgment, indent=2))
                print(f"\nwritten -> {out_path}")
                # exit code 0 = passed, 1 = failed; useful for the loop driver
                sys.exit(0 if judgment.get("passed") else 1)
            except json.JSONDecodeError as e:
                last_err = f"JSON parse failed for {m}: {e}\nRaw: {text!r}"
            except Exception as e:
                last_err = f"{m}: {type(e).__name__}: {e}"
                print(f"[judge] {last_err}")
                continue
        raise SystemExit(f"all model attempts failed; last: {last_err}")
    finally:
        try: os.unlink(mono_path)
        except Exception: pass


if __name__ == "__main__":
    main()
