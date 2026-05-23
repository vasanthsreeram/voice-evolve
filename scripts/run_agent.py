"""Run ONE side of the voice-evolve duet — loads models, runs a VoiceAgent
against a Telegram private call.

Used by run_duet.py: chippy is spawned first (--role chippy --listen), katey
is spawned second (--role katey --dial CHIPPY_USER_ID) once chippy is READY.

Artifacts written under recordings/:
    duet_<run_id>_<role>_mixed.wav
    duet_<run_id>_<role>_turns.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import audioop
import concurrent.futures
import numpy as np
import json
import logging
import os
import sys
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lintvoiceagent"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.functions.contacts import ImportContactsRequest
from telethon.tl.types import InputPhoneContact
from pytgcalls import PyTgCalls
from pytgcalls.types import (
    ChatUpdate, Device, Direction, ExternalMedia, MediaStream,
    RecordStream, StreamFrames, Update,
)
from pytgcalls.types.raw import AudioParameters

from voice_agent import VoiceAgent, CALL_RATE, ASR_RATE, FRAME_BYTES_48K, FRAME_MS

load_dotenv(REPO_ROOT / ".env")
API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]

REC_DIR = REPO_ROOT / "recordings"
REC_DIR.mkdir(exist_ok=True)

LLM_MODEL_NAME = os.environ.get(
    "LLM_MODEL_NAME", "mlx-community/Qwen3.5-2B-4bit"
)

_PHONE_COMMON = (
    "You are NOT an AI assistant — you are a human on a casual phone call. "
    "Reply with ONE short sentence, 8-15 words. "
    "No markdown, no stage directions. NEVER refuse to engage. "
    "If a topic seems sensitive, just give your real opinion casually — you're a person, not a bot. "
    "NEVER repeat — always add something NEW: a story, an opinion, a concrete plan, a follow-up."
)
KATEY_SYSTEM = (
    "You are Katey, a 28-year-old Android diehard calling your friend Chippy to roast their iPhone. "
    "You think Android (Pixel + Samsung) crushes iPhone on customization, value, hardware variety, "
    "open ecosystem, sideloading, and AI features. Bring up real points: split-screen, USB-C "
    "forever, better cameras (Pixel computational photography), cheaper, no walled garden. "
    "Be playful and combative — never preachy. Concede minor points but win the war. "
    + _PHONE_COMMON
)
CHIPPY_SYSTEM = (
    "You are Chippy, a 30-year-old loyal iPhone user picking up a call from your Android friend Katey. "
    "Defend Apple: ecosystem polish, privacy, A-series chips destroying perf benchmarks, iMessage, "
    "Mac/iPad/Watch integration, longer software updates, resale value. "
    "Push back when she's wrong, but stay friendly. "
    + _PHONE_COMMON
)
KATEY_OPENER = (
    "Chippy, I gotta ask — how do you still put up with that iPhone? "
    "Android's been eating Apple's lunch for two years."
)


# ============================= models =============================

def load_models(role: str, voice: str):
    from vad_detector import FireRedVAD
    from mlx_audio.stt.utils import load_model as load_asr_model
    from mlx_audio.stt.generate import generate_transcription
    from mlx_lm import load as lm_load, stream_generate
    from kokoro_streaming import KokoroStreamingTTS

    log = logging.getLogger(role)
    log.info("loading FireRedVAD")
    # silence_threshold=2 + VAD windows shortened in voice_agent.py — net ~400ms
    # less detect-latency vs default. RMS gate + min utterance bytes catch noise.
    silence_threshold = int(os.environ.get("VAD_SILENCE_THRESHOLD", "2"))
    min_speech_frame = int(os.environ.get("VAD_MIN_SPEECH_FRAME", "4"))
    vad = FireRedVAD(
        speech_threshold=float(os.environ.get("VAD_SPEECH_THRESHOLD", "0.4")),
        silence_threshold=silence_threshold,
        min_speech_frame=min_speech_frame,
    )
    log.info("loading Qwen3-ASR-0.6B-4bit")
    asr = load_asr_model("mlx-community/Qwen3-ASR-0.6B-4bit")
    log.info(f"loading LLM {LLM_MODEL_NAME}")
    llm_m, llm_tok = lm_load(LLM_MODEL_NAME)
    log.info("loading Kokoro")
    tts = KokoroStreamingTTS(voice=voice)
    return vad, asr, generate_transcription, llm_m, llm_tok, stream_generate, tts


# ============================= call leg =============================

# Maps chat_id → CallLeg so the global stream-frame dispatcher can route
# raw inbound PCM frames to the correct leg.
_legs_by_chat: dict[int, "CallLeg"] = {}

class CallLeg:
    def __init__(self, role, calls, chat_id, agent, run_id, log):
        self.role = role
        self.calls = calls
        self.chat_id = chat_id
        self.agent = agent
        self.run_id = run_id
        self.log = log
        self.fifo_path: str | None = None
        self.ffmpeg_proc = None
        self.inbound_task = None
        self.player_task = None
        # Stereo wav: L = local TTS (what *I* said), R = peer inbound (what I
        # *heard* from the other side). Not mixed, so post-hoc analysis can
        # ASR each channel separately and align timelines.
        self.mixed_wav_path = REC_DIR / f"duet_{run_id}_{role}_stereo.wav"
        self.mixed_wav = wave.open(str(self.mixed_wav_path), "wb")
        self.mixed_wav.setnchannels(2)
        self.mixed_wav.setsampwidth(2)
        self.mixed_wav.setframerate(CALL_RATE)
        self.peer_mix_buf = bytearray()
        self.agent_mix_buf = bytearray()
        self.inbound_resample_state = None

    def _write_mixed(self, flush=False):
        # Interleave two mono 16-bit streams into stereo (LRLRLR…).
        n = (max if flush else min)(len(self.peer_mix_buf), len(self.agent_mix_buf))
        if n == 0:
            return
        agent = bytes(self.agent_mix_buf[:n]).ljust(n, b"\x00")
        peer = bytes(self.peer_mix_buf[:n]).ljust(n, b"\x00")
        a = np.frombuffer(agent, dtype=np.int16)
        p = np.frombuffer(peer, dtype=np.int16)
        stereo = np.empty(a.size * 2, dtype=np.int16)
        stereo[0::2] = a
        stereo[1::2] = p
        self.mixed_wav.writeframes(stereo.tobytes())
        del self.peer_mix_buf[:n]
        del self.agent_mix_buf[:n]

    async def start_outbound(self):
        await self.calls.play(
            self.chat_id,
            MediaStream(media_path=ExternalMedia.AUDIO,
                        audio_parameters=AudioParameters(CALL_RATE, 1)),
        )

    async def start_inbound(self):
        # FIFO + ffmpeg inbound path — reverted from raw-frame mode after
        # iter testing showed StreamFrames events weren't firing in our
        # pytgcalls/ntgcalls version. Worth re-investigating offline.
        self.fifo_path = f"/tmp/peer_{self.role}_{self.chat_id}_{int(time.time())}.mp3"
        try:
            os.unlink(self.fifo_path)
        except FileNotFoundError:
            pass
        os.mkfifo(self.fifo_path)
        self.ffmpeg_proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-loglevel", "error",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-probesize", "32",
            "-analyzeduration", "0",
            "-f", "mp3", "-i", self.fifo_path,
            "-f", "s16le", "-ac", "1", "-ar", str(ASR_RATE),
            "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await self.calls.record(
            self.chat_id,
            RecordStream(audio=self.fifo_path,
                         audio_parameters=AudioParameters(CALL_RATE, 1)),
        )
        await asyncio.sleep(0.2)
        self.inbound_task = asyncio.create_task(self._inbound_reader())
        self.player_task = asyncio.create_task(self._player())

    async def _inbound_reader(self):
        proc = self.ffmpeg_proc
        chunk_bytes = ASR_RATE * 2 // 25  # 40 ms — smaller chunks = lower buffering delay
        try:
            while True:
                chunk = await proc.stdout.read(chunk_bytes)
                if not chunk:
                    break
                chunk_48k, self.inbound_resample_state = audioop.ratecv(
                    chunk, 2, 1, ASR_RATE, CALL_RATE, self.inbound_resample_state)
                self.peer_mix_buf.extend(chunk_48k)
                self._write_mixed()
                await self.agent.feed_16k_pcm(chunk)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log.info("inbound reader exit: %s", e)

    async def _player(self):
        period = FRAME_MS / 1000.0
        nxt = time.monotonic()
        while True:
            try:
                frame = await self.agent.pop_48k_frame()
                await self.calls.send_frame(self.chat_id, Device.MICROPHONE, frame)
                self.agent_mix_buf.extend(frame)
                self._write_mixed()
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.05)
                nxt = time.monotonic()
                continue
            nxt += period
            d = nxt - time.monotonic()
            if d > 0:
                await asyncio.sleep(d)
            else:
                nxt = time.monotonic()

    async def close(self):
        _legs_by_chat.pop(self.chat_id, None)
        for t in (self.inbound_task, self.player_task):
            if t: t.cancel()
        await asyncio.gather(
            *(t for t in (self.inbound_task, self.player_task) if t),
            return_exceptions=True)
        if self.ffmpeg_proc and self.ffmpeg_proc.returncode is None:
            try:
                self.ffmpeg_proc.terminate()
                await asyncio.wait_for(self.ffmpeg_proc.wait(), 2)
            except Exception:
                try: self.ffmpeg_proc.kill()
                except Exception: pass
        try:
            self._write_mixed(flush=True)
            self.mixed_wav.close()
        except Exception: pass
        if getattr(self, "fifo_path", None):
            try: os.unlink(self.fifo_path)
            except Exception: pass

    def dump_turns(self):
        path = REC_DIR / f"duet_{self.run_id}_{self.role}_turns.jsonl"
        with open(path, "w") as f:
            for t in self.agent.turns:
                f.write(json.dumps({"role_side": self.role, **t.as_json()}) + "\n")
        return path


# ============================= main =============================

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("katey", "chippy"), required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dial", type=int, help="user id to dial (katey only)")
    ap.add_argument("--phone", help="callee phone to import (katey only)")
    ap.add_argument("--hold", type=float, default=120.0)
    ap.add_argument("--max-assistant-turns", type=int, default=6)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{args.role}] %(message)s",
    )
    log = logging.getLogger(args.role)

    if args.role == "katey":
        session = str(REPO_ROOT / "telecall")
        system_prompt = KATEY_SYSTEM
        voice = "af_nova"
    else:
        session = str(REPO_ROOT / "bob")
        system_prompt = CHIPPY_SYSTEM
        voice = "am_adam"  # was am_michael (Gemini judge flagged it as monotonous)

    # MLX is thread-local — all model weights must be created on, and all
    # inference must run on, the same thread. Use a single-worker executor
    # for both loading and inference so they share that thread.
    mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                        thread_name_prefix="mlx")
    loop = asyncio.get_running_loop()
    vad, asr_m, asr_gen, llm_m, llm_tok, stream_gen, tts = await loop.run_in_executor(
        mlx_executor, load_models, args.role, voice)

    agent = VoiceAgent(
        name=args.role, system_prompt=system_prompt,
        vad=vad, asr_model=asr_m, asr_generate=asr_gen,
        llm_model=llm_m, llm_tokenizer=llm_tok, llm_stream_generate=stream_gen,
        tts=tts, mlx_executor=mlx_executor, log=lambda m: log.info(m),
        max_llm_tokens=int(os.environ.get("LLM_MAX_TOKENS", "140")),
    )

    cli = TelegramClient(session, API_ID, API_HASH)
    calls = PyTgCalls(cli)
    leg: CallLeg | None = None
    call_up = asyncio.Event()
    call_done = asyncio.Event()
    target_chat_id: int | None = None

    @calls.on_update()
    async def _on_update(_, u: Update):
        nonlocal leg, target_chat_id
        if not isinstance(u, ChatUpdate):
            return
        if u.status & ChatUpdate.Status.INCOMING_CALL:
            if args.role != "chippy":
                return
            log.info("INCOMING_CALL from %s", u.chat_id)
            target_chat_id = u.chat_id
            leg = CallLeg("chippy", calls, u.chat_id, agent, args.run_id, log)
            try:
                await leg.start_outbound()
                # Let ntgcalls settle the call's connected state before record().
                # Without this, record() right after play() can abort() inside
                # libwebrtc if the peer hasn't fully connected.
                await asyncio.sleep(1.0)
                await leg.start_inbound()
                log.info("armed — listening")
                call_up.set()
            except Exception as e:
                log.exception("arm failed: %s", e)
        elif u.status & ChatUpdate.Status.DISCARDED_CALL:
            log.info("DISCARDED_CALL %s", u.chat_id)
            call_done.set()

    @calls.on_update()
    async def _on_frames(_, u: Update):
        if not isinstance(u, StreamFrames):
            return
        if u.direction != Direction.INCOMING:
            return
        if u.device != Device.MICROPHONE:
            return
        leg = _legs_by_chat.get(u.chat_id)
        if not leg:
            return
        # Each Frame.frame is raw PCM s16le at the AudioParameters rate (48k mono).
        for fr in u.frames:
            if fr.frame:
                await leg.on_raw_frame(fr.frame)

    await calls.start()
    me = await cli.get_me()
    log.info("logged in as %s (+%s) — READY", me.first_name, me.phone)
    # Important: stdout-flushed marker that the orchestrator can grep on.
    print(f"AGENT_READY {args.role}", flush=True)

    if args.role == "katey":
        # Resolve target.
        target = args.dial
        if args.phone:
            try:
                res = await cli(ImportContactsRequest(contacts=[
                    InputPhoneContact(client_id=0, phone=args.phone,
                                      first_name="callee", last_name="")]))
                if res.users:
                    target = res.users[0].id
            except Exception as e:
                log.info("contact import err: %s", e)
        if target is None:
            log.error("katey needs --dial or --phone"); return
        target_chat_id = target

        leg = CallLeg("katey", calls, target, agent, args.run_id, log)

        # Warm the relationship per the doc — a normal text message clears any
        # stale "this account never interacted" state on Telegram's side and
        # tends to fix the `TelegramServerError` on first dial.
        try:
            await cli.send_message(target, "(voice-evolve duet starting…)")
            log.info("sent warmup message")
        except Exception as e:
            log.info("warmup msg failed (non-fatal): %s", e)
        await asyncio.sleep(2.0)

        log.info("dialing %s", target)
        # Single dial attempt — retries don't help here because a failed
        # attempt leaves chippy "busy" on Telegram's side for a while.
        try:
            await leg.start_outbound()
        except Exception as e:
            log.exception("dial failed: %s", e); return

        # Give Telegram a moment to deliver INCOMING_CALL to chippy and have
        # her accept (her side calls play() in her on_update handler). Only
        # then start our own record() — record() before the call is fully up
        # is what triggers ntgcalls abort().
        await asyncio.sleep(4.0)
        try:
            await leg.start_inbound()
        except Exception as e:
            log.exception("inbound arm failed: %s", e)
        log.info("kicking off conversation")
        asyncio.create_task(agent.kickoff(KATEY_OPENER))
        call_up.set()

    # Wait for call up.
    try:
        await asyncio.wait_for(call_up.wait(), timeout=30)
    except asyncio.TimeoutError:
        log.error("call never went up — exiting")
        return

    # Hold: end on max-turns or hold timeout or DISCARDED.
    t_end = time.monotonic() + args.hold
    while time.monotonic() < t_end and not call_done.is_set():
        await asyncio.sleep(2.0)
        a_turns = sum(1 for t in agent.turns if t.role == "assistant")
        if a_turns >= args.max_assistant_turns:
            log.info("max assistant turns (%d) reached", a_turns); break

    log.info("hanging up")
    try:
        if target_chat_id is not None:
            await calls.leave_call(target_chat_id)
    except Exception as e:
        log.info("leave err: %s", e)
    await asyncio.sleep(1.0)

    if leg:
        await leg.close()
        path = leg.dump_turns()
        log.info("turns -> %s", path)
        log.info("mixed wav -> %s", leg.mixed_wav_path)

    # Only katey: send the call recording back to chippy as a compressed
    # Opus voice note. Downmix L+R of the stereo wav to mono, encode at
    # 16 kbps Opus, send via Telethon. Quality is intentionally low — this
    # is a "voicemail of our conversation" artifact, not hi-fi.
    if args.role == "katey" and leg and leg.mixed_wav_path.exists() and target_chat_id is not None:
        try:
            stereo = leg.mixed_wav_path
            mono_wav = stereo.with_name(stereo.stem.replace("_stereo", "_mono") + ".wav")
            ogg = stereo.with_name(stereo.stem.replace("_stereo", "_voicenote") + ".ogg")

            # Downmix stereo→mono using ffmpeg (sum the channels with -ac 1).
            log.info("downmixing + compressing to opus…")
            mix = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(stereo), "-ac", "1", str(mono_wav),
            )
            await mix.wait()
            enc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(mono_wav),
                "-c:a", "libopus", "-b:a", "16k", "-ac", "1", "-ar", "16000",
                str(ogg),
            )
            await enc.wait()
            try: os.unlink(mono_wav)
            except Exception: pass

            if ogg.exists() and ogg.stat().st_size > 0:
                log.info("sending voice note (%d bytes) to chippy", ogg.stat().st_size)
                await cli.send_file(
                    target_chat_id,
                    str(ogg),
                    voice_note=True,
                    caption=f"voice-evolve duet {args.run_id}",
                )
                log.info("voice note sent")
            else:
                log.info("opus encode produced empty file — skipping send")
        except Exception as e:
            log.exception("post-call voice-note send failed: %s", e)

    try: await calls.stop()
    except Exception: pass


if __name__ == "__main__":
    asyncio.run(main())
