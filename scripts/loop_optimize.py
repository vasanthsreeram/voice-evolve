"""Outer optimization loop: run duet → ask Gemini to judge → if not passed,
nudge the tunable knobs based on the judgment's recommendations, then run again.

Knobs are exposed as env vars consumed by run_agent.py / voice_agent.py:
    VAD_SILENCE_THRESHOLD     (int, default 2; lower = faster but more false turn-ends)
    VAD_MIN_SPEECH_FRAME      (int, default 4)
    TTS_TAIL_GUARD_S          (float, default 0.3; lower = faster, more echo risk)
    INITIAL_DEAFEN_S          (float, default 1.0)
    MIN_UTTERANCE_BYTES       (int, default 48000; lower = more responsive, more junk)
    MIN_UTTERANCE_RMS         (float, default 0.005)
    LLM_MAX_TOKENS            (int, default 140)
    KOKORO_STREAM_INTERVAL    (float, default 0.3)

Stops when Gemini returns passed=true OR --max-iters is hit.

Each iteration's full config + Gemini judgment goes into
    recordings/loop_<loop_id>/iter_<n>.json
plus the underlying duet artifacts under recordings/duet_<run_id>_*.

Usage:
    cd /Users/vas/Documents/voice-evolve
    set -a && . ./.env && set +a
    uv run --project lintvoiceagent python scripts/loop_optimize.py \\
        --hold 120 --max-iters 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REC_DIR = REPO_ROOT / "recordings"


# Starting config — calibrated from loop_20260522T191338Z iter 2 with Qwen 0.8B-8bit
# (best Gemini score 5.75: pacing 6, overlap 10, coherence 8, naturalness 7, latency 3.0s).
DEFAULT_KNOBS = {
    "VAD_SILENCE_THRESHOLD": 2,      # tried 1, FireRedVAD became unreliable → reverted
    "VAD_MIN_SPEECH_FRAME": 4,
    "TTS_TAIL_GUARD_S": 0.122,
    "INITIAL_DEAFEN_S": 0.48,
    "MIN_UTTERANCE_BYTES": 48000,
    "MIN_UTTERANCE_RMS": 0.005,
    "LLM_MAX_TOKENS": 78,
    "KOKORO_STREAM_INTERVAL": 0.3,
}

# Clamp ranges so we don't drift into broken territory.
KNOB_BOUNDS = {
    "VAD_SILENCE_THRESHOLD": (1, 6),
    "VAD_MIN_SPEECH_FRAME": (2, 8),
    "TTS_TAIL_GUARD_S": (0.1, 1.5),
    "INITIAL_DEAFEN_S": (0.3, 3.0),
    "MIN_UTTERANCE_BYTES": (16000, 96000),
    "MIN_UTTERANCE_RMS": (0.002, 0.02),
    "LLM_MAX_TOKENS": (60, 220),
    "KOKORO_STREAM_INTERVAL": (0.15, 0.4),
}


def clamp(name, val):
    lo, hi = KNOB_BOUNDS[name]
    if isinstance(DEFAULT_KNOBS[name], int):
        return max(lo, min(hi, int(round(val))))
    return max(lo, min(hi, float(val)))


def score(judgment: dict) -> float:
    """Composite quality score for comparing iterations.

    Higher = better. Combines Gemini's qualitative scores AND latency, since
    the user wants both fluency and speed tracked via Gemini's feedback.
    """
    if not judgment:
        return -1e9
    pacing = judgment.get("pacing_score", 0)
    overlap = judgment.get("overlap_score", 0)
    coherence = judgment.get("coherence_score", 0)
    naturalness = judgment.get("naturalness_score", 0)
    latency = float(judgment.get("observed_response_latency_s") or 0)
    # Quality side: average of the four 0-10 scores (so max 10).
    quality = (pacing + overlap + coherence + naturalness) / 4.0
    # Latency penalty: 0 at <=1s, -3 at 4s, -6 at 7s+. Caps the damage so a
    # great-quality call still wins over a fast but broken one.
    if latency <= 1.0:
        lat_pen = 0
    elif latency >= 7.0:
        lat_pen = 6
    else:
        lat_pen = (latency - 1.0) * 1.0   # 1 point lost per second past 1s
    return quality - lat_pen


def decide_knob_changes(knobs: dict, judgment: dict) -> tuple[dict, list[str]]:
    """Map Gemini's descriptive scores+issues to ONE small knob nudge.

    Picks the single most actionable symptom and applies a small (5-15%) move
    in the right direction. Loop driver handles best-so-far rollback when the
    nudge backfires.
    """
    new = dict(knobs)
    rationale: list[str] = []

    pacing = judgment.get("pacing_score", 0)
    overlap = judgment.get("overlap_score", 10)
    coherence = judgment.get("coherence_score", 10)
    naturalness = judgment.get("naturalness_score", 10)
    latency = float(judgment.get("observed_response_latency_s") or 0)
    issues = " ".join(judgment.get("issues", [])).lower()

    # Priority order: fix overlap/truncation first (correctness), then latency,
    # then quality knobs. Only ONE adjustment per iteration to keep moves small.
    if overlap <= 6 or any(
        kw in issues for kw in ("overlap", "talk over", "interrupt", "simultaneous")):
        new["TTS_TAIL_GUARD_S"] = clamp("TTS_TAIL_GUARD_S", knobs["TTS_TAIL_GUARD_S"] * 1.15)
        new["VAD_SILENCE_THRESHOLD"] = clamp("VAD_SILENCE_THRESHOLD",
                                              knobs["VAD_SILENCE_THRESHOLD"] + 1)
        rationale.append(f"overlap={overlap} → +15% tail-guard, VAD silence +1")
    elif any(kw in issues for kw in ("cut off", "truncated", "mid-sentence", "incomplete")):
        new["VAD_SILENCE_THRESHOLD"] = clamp("VAD_SILENCE_THRESHOLD",
                                              knobs["VAD_SILENCE_THRESHOLD"] + 1)
        rationale.append("speech truncated → VAD silence +1")
    elif any(kw in issues for kw in (
            "hallucinat", "the.", "thank you", "random phrase", "out of context")):
        new["MIN_UTTERANCE_BYTES"] = clamp("MIN_UTTERANCE_BYTES",
                                            knobs["MIN_UTTERANCE_BYTES"] * 1.15)
        rationale.append("junk ASR turns → +15% min utterance bytes")
    elif latency > 2.5 or pacing <= 6 or any(
        kw in issues for kw in ("silence", "long gap", "slow", "delay")):
        # Small nudge — never aggressive.
        new["TTS_TAIL_GUARD_S"] = clamp("TTS_TAIL_GUARD_S", knobs["TTS_TAIL_GUARD_S"] * 0.88)
        new["INITIAL_DEAFEN_S"] = clamp("INITIAL_DEAFEN_S", knobs["INITIAL_DEAFEN_S"] * 0.9)
        new["LLM_MAX_TOKENS"] = clamp("LLM_MAX_TOKENS", knobs["LLM_MAX_TOKENS"] * 0.92)
        rationale.append(
            f"latency={latency:.1f}s pacing={pacing} → -12% tail-guard, "
            f"-10% deafen, -8% LLM tokens")
    elif any(kw in issues for kw in ("long-winded", "rambling", "too long", "verbose")):
        new["LLM_MAX_TOKENS"] = clamp("LLM_MAX_TOKENS", knobs["LLM_MAX_TOKENS"] * 0.8)
        rationale.append("verbose replies → -20% LLM tokens")
    elif naturalness <= 4 or any(kw in issues for kw in ("robotic", "monotone", "synthetic")):
        rationale.append("voice naturalness low — not auto-tunable (TTS model lever)")
    elif coherence <= 5 or any(kw in issues for kw in ("ignore", "irrelevant", "off-topic",
                                                       "doesn't respond")):
        rationale.append("coherence weak — persona prompt edit is the lever (manual)")
    else:
        rationale.append("no clear actionable signal — leaving knobs unchanged")

    return new, rationale


def run_duet(env_overrides: dict, hold_s: float) -> str | None:
    """Spawn run_duet.py with the given env. Returns the run_id on success."""
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    env["PYTHONUNBUFFERED"] = "1"
    # Pin the LLM here so we don't have to set it externally.
    env.setdefault(
        "LLM_MODEL_NAME",
        str(Path.home() / ".lmstudio/models/mlx-community/Qwen3.5-4B-MLX-8bit"),
    )
    # Wait so Telegram's stale-call state clears between runs.
    cooldown = int(os.environ.get("LOOP_COOLDOWN_S", "75"))
    print(f"[loop] cooldown {cooldown}s before next call…")
    time.sleep(cooldown)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_duet.py"),
        "--hold", str(hold_s),
        "--max-turns", "999",
    ]
    print(f"[loop] launching duet with knobs: "
          f"{json.dumps({k: env_overrides.get(k) for k in DEFAULT_KNOBS}, indent=0)}")
    log_path = REC_DIR / f"loop_duet_{int(time.time())}.log"
    proc = subprocess.run(cmd, stdout=open(log_path, "wb"),
                          stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))
    print(f"[loop] duet exit={proc.returncode}  log={log_path}")
    # Pull the run_id from the log.
    with open(log_path) as f:
        for line in f:
            if line.startswith("=== voice-evolve duet  run_id="):
                run_id = line.strip().split("run_id=")[1].rstrip(" =")
                return run_id
    return None


def run_judge(run_id: str) -> dict | None:
    print(f"[loop] judging run {run_id}…")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "judge_call.py"),
        "--run-id", run_id,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env=os.environ.copy(), cwd=str(REPO_ROOT))
    print(proc.stdout)
    if proc.stderr:
        print("[loop][stderr]", proc.stderr)
    judgment_path = REC_DIR / f"duet_{run_id}_judgment.json"
    if judgment_path.exists():
        return json.loads(judgment_path.read_text())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=float, default=120.0)
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--start-from-defaults", action="store_true",
                    help="ignore any KNOB env vars already set, start from DEFAULT_KNOBS")
    args = ap.parse_args()

    loop_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    loop_dir = REC_DIR / f"loop_{loop_id}"
    loop_dir.mkdir(exist_ok=True)
    print(f"[loop] id={loop_id}  dir={loop_dir}")

    knobs = dict(DEFAULT_KNOBS) if args.start_from_defaults else {
        k: type(v)(os.environ.get(k, v)) for k, v in DEFAULT_KNOBS.items()
    }

    # Track the best knob set so we can roll back if a nudge makes things worse.
    best_knobs = dict(knobs)
    best_score = -1e9
    best_iter = 0

    history = []
    for it in range(1, args.max_iters + 1):
        print(f"\n========== ITERATION {it}/{args.max_iters} ==========")
        print(f"[loop] knobs: {knobs}")

        run_id = run_duet(knobs, args.hold)
        if not run_id:
            print("[loop] duet did not produce a run_id — aborting")
            break

        # Verify stereo wavs exist.
        if not (REC_DIR / f"duet_{run_id}_katey_stereo.wav").exists():
            print(f"[loop] missing katey stereo wav for {run_id} — bad run, retrying")
            history.append({"iter": it, "run_id": run_id, "judgment": None,
                            "knobs": dict(knobs), "error": "no_wav"})
            continue

        judgment = run_judge(run_id)
        history.append({"iter": it, "run_id": run_id, "judgment": judgment,
                        "knobs": dict(knobs)})

        with open(loop_dir / f"iter_{it:02d}.json", "w") as f:
            json.dump({"iter": it, "run_id": run_id, "judgment": judgment,
                       "knobs": knobs}, f, indent=2)

        if judgment is None:
            print("[loop] judgment failed — keeping knobs and retrying")
            continue

        cur_score = score(judgment)
        print(f"[loop] composite score: {cur_score:.2f}  (best so far: {best_score:.2f} @ iter {best_iter})")

        if judgment.get("passed"):
            print(f"\n🎉 [loop] PASSED at iteration {it}!")
            print(f"    overall scores: pacing={judgment.get('pacing_score')} "
                  f"overlap={judgment.get('overlap_score')} "
                  f"coherence={judgment.get('coherence_score')} "
                  f"naturalness={judgment.get('naturalness_score')} "
                  f"latency_s={judgment.get('observed_response_latency_s')}")
            print(f"    final knobs: {knobs}")
            break

        # Network-glitch detection: if latency is wildly out of expected range
        # (> 30s), the call essentially didn't work — don't blame the knobs,
        # just retry with the SAME knobs. Applies even on the first iteration
        # (a 110-second silence is never legit).
        glitch_latency = float(judgment.get("observed_response_latency_s") or 0)
        if glitch_latency > 30:
            print(f"[loop] suspected network glitch (latency={glitch_latency:.0f}s) "
                  f"— same knobs, retrying without scoring")
            continue

        # Rollback if this iteration regressed vs the best so far.
        if cur_score < best_score - 1.5:
            print(f"[loop] REGRESSION (this {cur_score:.2f} vs best {best_score:.2f}) "
                  f"— reverting to best knobs from iter {best_iter} and trying a different lever")
            # Revert and pick a NEW direction: bias against repeating the last move.
            knobs = dict(best_knobs)
            # Apply a perturbation that's a slight loosening (less aggressive than last try).
            knobs["TTS_TAIL_GUARD_S"] = clamp("TTS_TAIL_GUARD_S",
                                               knobs["TTS_TAIL_GUARD_S"] * 1.08)
            print(f"[loop] reverted knobs: {knobs}")
            continue

        if cur_score > best_score:
            best_score = cur_score
            best_knobs = dict(knobs)
            best_iter = it
            print(f"[loop] new best (score {cur_score:.2f})")

        new_knobs, rationale = decide_knob_changes(knobs, judgment)
        print(f"[loop] gemini issues: {judgment.get('issues')}")
        print(f"[loop] decisions:")
        for r in rationale:
            print(f"        - {r}")
        diff = {k: (knobs[k], new_knobs[k]) for k in knobs if knobs[k] != new_knobs[k]}
        if diff:
            print(f"[loop] knob changes: " + ", ".join(
                f"{k}: {old}→{new}" for k, (old, new) in diff.items()))
        else:
            print("[loop] no knob changes — same config will be re-judged")
        knobs = new_knobs
    else:
        print(f"\n[loop] max iterations ({args.max_iters}) reached without passing.")

    summary = {"loop_id": loop_id, "iterations": history}
    with open(loop_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[loop] summary -> {loop_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
