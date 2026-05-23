"""Orchestrate a katey↔chippy voice-to-voice Telegram call.

Spawns two `run_agent.py` subprocesses because ntgcalls / libwebrtc abort()s
when two PyTgCalls instances live in the same process. Chippy comes up first
(listener), then katey dials her once chippy prints AGENT_READY.

After both processes exit, merges per-side turns.jsonl into a single
chronological convo transcript so we can read whether the call sounded
like a normal conversation.

Usage:
    cd /Users/vas/Documents/voice-evolve
    set -a && . ./.env && set +a
    uv run --project lintvoiceagent python scripts/run_duet.py --hold 120
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REC_DIR = REPO_ROOT / "recordings"
REC_DIR.mkdir(exist_ok=True)

CHIPPY_PHONE = "+6588461730"
CHIPPY_USER_ID = "8943154461"


async def stream_lines(proc: asyncio.subprocess.Process, prefix: str,
                       ready_evt: asyncio.Event, log_path: Path):
    """Tee subprocess stdout: print live, save to log file, fire on AGENT_READY."""
    log_f = open(log_path, "wb")
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            log_f.write(line)
            log_f.flush()
            try:
                s = line.decode(errors="replace").rstrip()
            except Exception:
                continue
            print(f"[{prefix}] {s}", flush=True)
            if not ready_evt.is_set() and s.startswith("AGENT_READY"):
                ready_evt.set()
    finally:
        log_f.close()


async def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=float, default=120.0)
    ap.add_argument("--max-turns", type=int, default=6)
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"=== voice-evolve duet  run_id={run_id} ===", flush=True)

    py = sys.executable
    script = str(REPO_ROOT / "scripts" / "run_agent.py")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Half-duplex floor coordination — both subprocesses share this lock file.
    # Only one of them holds the exclusive flock while its TTS is playing,
    # so the other can't talk over it. Path is per-run so concurrent runs
    # don't collide.
    floor_path = f"/tmp/duet_floor_{run_id}.lock"
    try: os.unlink(floor_path)
    except FileNotFoundError: pass
    env["DUET_FLOOR_PATH"] = floor_path

    # ----- spawn chippy (listener) first -----
    chippy_log = REC_DIR / f"duet_{run_id}_chippy.log"
    chippy_proc = await asyncio.create_subprocess_exec(
        py, script, "--role", "chippy", "--run-id", run_id,
        "--hold", str(args.hold), "--max-assistant-turns", str(args.max_turns),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        env=env, cwd=str(REPO_ROOT),
    )
    chippy_ready = asyncio.Event()
    chippy_streamer = asyncio.create_task(
        stream_lines(chippy_proc, "chippy", chippy_ready, chippy_log)
    )
    print(f"chippy pid={chippy_proc.pid} — waiting for AGENT_READY (model load ~30s)…",
          flush=True)
    try:
        await asyncio.wait_for(chippy_ready.wait(), timeout=180)
    except asyncio.TimeoutError:
        print("chippy never reported READY — aborting", flush=True)
        chippy_proc.terminate()
        await asyncio.gather(chippy_streamer, chippy_proc.wait(),
                             return_exceptions=True)
        return

    # ----- spawn katey (initiator) -----
    katey_log = REC_DIR / f"duet_{run_id}_katey.log"
    katey_proc = await asyncio.create_subprocess_exec(
        py, script, "--role", "katey", "--run-id", run_id,
        "--phone", CHIPPY_PHONE, "--dial", CHIPPY_USER_ID,
        "--hold", str(args.hold), "--max-assistant-turns", str(args.max_turns),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        env=env, cwd=str(REPO_ROOT),
    )
    katey_ready = asyncio.Event()
    katey_streamer = asyncio.create_task(
        stream_lines(katey_proc, "katey", katey_ready, katey_log)
    )
    print(f"katey pid={katey_proc.pid} — model load + dial…", flush=True)

    # Wait for both processes to exit.
    rc_c, rc_k, _, _ = await asyncio.gather(
        chippy_proc.wait(), katey_proc.wait(),
        chippy_streamer, katey_streamer,
        return_exceptions=True,
    )
    print(f"both processes exited (chippy rc={rc_c}, katey rc={rc_k})", flush=True)

    # ----- merge transcripts -----
    convo_path = REC_DIR / f"duet_{run_id}_convo.txt"
    turns = []
    for side in ("katey", "chippy"):
        p = REC_DIR / f"duet_{run_id}_{side}_turns.jsonl"
        if not p.exists():
            print(f"WARN no turns file for {side}: {p}", flush=True); continue
        with open(p) as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                turns.append(j)
    turns.sort(key=lambda j: j.get("t", 0))
    with open(convo_path, "w") as f:
        f.write(f"# duet run {run_id}\n\n")
        for j in turns:
            if j.get("role") != "assistant":
                continue
            ts = datetime.fromtimestamp(j["t"], timezone.utc).strftime("%H:%M:%S")
            f.write(f"[{ts}] {j['role_side']}: {j['text']}\n")
    print(f"\n=== CONVO TRANSCRIPT  ({convo_path}) ===", flush=True)
    try:
        with open(convo_path) as f:
            sys.stdout.write(f.read())
    except Exception as e:
        print(f"could not read convo file: {e}", flush=True)

    print(f"\nmixed wavs:", flush=True)
    for side in ("katey", "chippy"):
        p = REC_DIR / f"duet_{run_id}_{side}_mixed.wav"
        if p.exists():
            sz = p.stat().st_size
            print(f"  {p}  ({sz:,} bytes)", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
