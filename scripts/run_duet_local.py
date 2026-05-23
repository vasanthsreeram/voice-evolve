"""Run a two-agent duet entirely in-process, bypassing Telegram.

Both VoiceAgents share the same MLX models (one worker thread for inference).
Katey's TTS PCM is piped into Chippy's `feed_16k_pcm`, and vice versa, via
in-memory async forwarders. This proves the local ASR→LLM→TTS pipeline is
capable of a sustained natural conversation when Telegram isn't dropping
audio out from under us.

Writes:
    recordings/local_<run_id>_convo.txt
    recordings/local_<run_id>_full.wav    a stereo mix: L=katey TTS, R=chippy TTS
    recordings/local_<run_id>_katey_turns.jsonl
    recordings/local_<run_id>_chippy_turns.jsonl

Usage:
    cd /Users/vas/Documents/voice-evolve
    set -a && . ./.env && set +a
    LLM_MODEL_NAME=mlx-community/Qwen3.5-4B-MLX-8bit \
      uv run --project lintvoiceagent python scripts/run_duet_local.py --hold 90
"""
from __future__ import annotations

import argparse
import asyncio
import audioop
import concurrent.futures
import json
import logging
import os
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lintvoiceagent"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from voice_agent import VoiceAgent, CALL_RATE, ASR_RATE, FRAME_BYTES_48K, FRAME_MS

REC_DIR = REPO_ROOT / "recordings"
REC_DIR.mkdir(exist_ok=True)

LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME",
                                "mlx-community/Qwen3.5-4B-MLX-8bit")

KATEY_SYSTEM = (
    "You are Katey, a 28-year-old who just called your friend Chippy to chat. "
    "You are NOT an AI assistant — you are a human on a casual phone call. "
    "Reply with ONE short sentence, 8-15 words. "
    "No markdown, no stage directions. "
    "NEVER repeat — always add something NEW: a story, an opinion, a concrete plan, a follow-up. "
    "Topics you might bring up: a new restaurant you tried, your weekend plans, a movie you watched."
)
CHIPPY_SYSTEM = (
    "You are Chippy, a 30-year-old who just picked up a call from your friend Katey. "
    "You are NOT an AI assistant — you are a human on a casual phone call. "
    "Reply with ONE short sentence, 8-15 words. "
    "No markdown, no stage directions. "
    "NEVER repeat — always add something NEW. "
    "React with curiosity and personal detail (your job, your weekend, what you ate). "
    "Ask one follow-up about HER specifics."
)
KATEY_OPENER = "Hey Chippy! It's Katey. How are you doing today?"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s")
log = logging.getLogger("local")


def load_models(mlx_executor: concurrent.futures.Executor):
    def _load():
        log.info("loading FireRedVAD")
        from vad_detector import FireRedVAD
        log.info("loading Qwen3-ASR-0.6B-4bit")
        from mlx_audio.stt.utils import load_model as load_asr_model
        from mlx_audio.stt.generate import generate_transcription
        log.info(f"loading LLM {LLM_MODEL_NAME}")
        from mlx_lm import load as lm_load, stream_generate
        log.info("loading Kokoro x2 (one per voice)")
        from kokoro_streaming import KokoroStreamingTTS

        vad_factory = lambda: FireRedVAD(
            speech_threshold=0.4,
            silence_threshold=int(os.environ.get("VAD_SILENCE_THRESHOLD", "2")),
            min_speech_frame=int(os.environ.get("VAD_MIN_SPEECH_FRAME", "4")),
        )
        asr = load_asr_model("mlx-community/Qwen3-ASR-0.6B-4bit")
        llm_m, llm_tok = lm_load(LLM_MODEL_NAME)
        tts_k = KokoroStreamingTTS(voice="af_nova")
        tts_c = KokoroStreamingTTS(voice="am_adam")
        return vad_factory, asr, generate_transcription, llm_m, llm_tok, stream_generate, tts_k, tts_c

    loop = asyncio.get_event_loop()
    return loop.run_in_executor(mlx_executor, _load)


class LocalAgent:
    """Owns a VoiceAgent + the stereo channel buffer for this side."""

    def __init__(self, role: str, voice_agent: VoiceAgent, log_):
        self.role = role
        self.agent = voice_agent
        self.log = log_
        # Capture our own TTS as 48k mono so we can both pipe it to the peer
        # AND save it for the final wav.
        self.tts_capture_buf = bytearray()
        self._lock = asyncio.Lock()

    async def player_loop(self, peer: "LocalAgent", stop_evt: asyncio.Event):
        """Pull 10ms frames from our TTS output and route to peer + log."""
        period = FRAME_MS / 1000.0
        nxt = time.monotonic()
        # Downsample state for piping into peer's 16k feed.
        rs48to16 = None
        while not stop_evt.is_set():
            frame = await self.agent.pop_48k_frame()  # 960 bytes
            self.tts_capture_buf.extend(frame)
            # Forward to peer's ASR pipeline (resample 48k → 16k mono).
            pcm16k, rs48to16 = audioop.ratecv(frame, 2, 1, CALL_RATE, ASR_RATE, rs48to16)
            try:
                await peer.agent.feed_16k_pcm(pcm16k)
            except Exception as e:
                self.log.info(f"[{self.role}] feed peer err: {e}")
            nxt += period
            d = nxt - time.monotonic()
            if d > 0:
                await asyncio.sleep(d)
            else:
                nxt = time.monotonic()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=float, default=90.0,
                    help="seconds to let the conversation run before hanging up")
    ap.add_argument("--max-turns", type=int, default=12,
                    help="hang up when EACH side has produced this many assistant turns")
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log.info(f"run_id={run_id}")

    mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                        thread_name_prefix="mlx")
    vad_factory, asr_m, asr_gen, llm_m, llm_tok, stream_gen, tts_k, tts_c = \
        await load_models(mlx_executor)

    # Half-duplex floor lock so only one agent speaks at a time.
    floor_path = f"/tmp/duet_floor_local_{run_id}.lock"
    try: os.unlink(floor_path)
    except FileNotFoundError: pass
    os.environ["DUET_FLOOR_PATH"] = floor_path

    katey_agent = VoiceAgent(
        name="katey", system_prompt=KATEY_SYSTEM,
        vad=vad_factory(), asr_model=asr_m, asr_generate=asr_gen,
        llm_model=llm_m, llm_tokenizer=llm_tok, llm_stream_generate=stream_gen,
        tts=tts_k, mlx_executor=mlx_executor,
        log=lambda m: log.info(m),
        max_llm_tokens=int(os.environ.get("LLM_MAX_TOKENS", "140")),
    )
    chippy_agent = VoiceAgent(
        name="chippy", system_prompt=CHIPPY_SYSTEM,
        vad=vad_factory(), asr_model=asr_m, asr_generate=asr_gen,
        llm_model=llm_m, llm_tokenizer=llm_tok, llm_stream_generate=stream_gen,
        tts=tts_c, mlx_executor=mlx_executor,
        log=lambda m: log.info(m),
        max_llm_tokens=int(os.environ.get("LLM_MAX_TOKENS", "140")),
    )

    katey = LocalAgent("katey", katey_agent, log)
    chippy = LocalAgent("chippy", chippy_agent, log)

    stop_evt = asyncio.Event()
    katey_task = asyncio.create_task(katey.player_loop(chippy, stop_evt))
    chippy_task = asyncio.create_task(chippy.player_loop(katey, stop_evt))

    log.info("kicking off conversation")
    asyncio.create_task(katey_agent.kickoff(KATEY_OPENER))

    # Watch turn counts; hang up when both hit max-turns OR --hold elapses.
    t_end = time.monotonic() + args.hold
    while time.monotonic() < t_end:
        await asyncio.sleep(2.0)
        k_turns = sum(1 for t in katey_agent.turns if t.role == "assistant")
        c_turns = sum(1 for t in chippy_agent.turns if t.role == "assistant")
        if k_turns >= args.max_turns and c_turns >= args.max_turns:
            log.info(f"max-turns reached on both sides (k={k_turns} c={c_turns}) — hanging up")
            break

    log.info("stopping…")
    stop_evt.set()
    await asyncio.gather(katey_task, chippy_task, return_exceptions=True)

    # Write stereo wav: L=katey_TTS, R=chippy_TTS (each in their own channel,
    # both on the same wall-clock timeline since they were generated live).
    n = min(len(katey.tts_capture_buf), len(chippy.tts_capture_buf))
    a = np.frombuffer(bytes(katey.tts_capture_buf[:n]), dtype=np.int16)
    b = np.frombuffer(bytes(chippy.tts_capture_buf[:n]), dtype=np.int16)
    stereo = np.empty(a.size * 2, dtype=np.int16)
    stereo[0::2] = a
    stereo[1::2] = b
    wav_path = REC_DIR / f"local_{run_id}_full.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(CALL_RATE)
        wf.writeframes(stereo.tobytes())
    log.info(f"wav -> {wav_path}")

    # Per-side turns jsonl.
    for side, ag in (("katey", katey_agent), ("chippy", chippy_agent)):
        p = REC_DIR / f"local_{run_id}_{side}_turns.jsonl"
        with open(p, "w") as f:
            for t in ag.turns:
                f.write(json.dumps({"role_side": side, **t.as_json()}) + "\n")
        log.info(f"turns -> {p}")

    # Merged convo.
    merged = []
    for t in katey_agent.turns:
        merged.append((t.t, "katey", t.role, t.text))
    for t in chippy_agent.turns:
        merged.append((t.t, "chippy", t.role, t.text))
    merged.sort(key=lambda r: r[0])
    convo_path = REC_DIR / f"local_{run_id}_convo.txt"
    with open(convo_path, "w") as f:
        f.write(f"# local duet {run_id}  llm={LLM_MODEL_NAME}\n\n")
        for ts, side, role, text in merged:
            if role != "assistant":
                continue
            stamp = datetime.fromtimestamp(ts, timezone.utc).strftime("%H:%M:%S")
            f.write(f"[{stamp}] {side}: {text}\n")
    log.info(f"convo -> {convo_path}")
    print(f"\n=== CONVO ({convo_path}) ===")
    print(open(convo_path).read())


if __name__ == "__main__":
    asyncio.run(main())
