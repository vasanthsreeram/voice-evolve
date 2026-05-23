"""Telegram private-call <-> Voice-Agent bridge.

Provider-agnostic: set VOICE_PROVIDER in .env to elevenlabs / openai / gemini.

Architecture:
  * Telethon signs in; pytgcalls auto-accepts incoming calls.
  * Outbound: provider emits PCM at provider.output_rate -> upsample to 48k ->
    pytgcalls.send_frame in 10 ms chunks.
  * Inbound: pytgcalls.record(RecordStream(audio="/tmp/peer_<id>.mp3")) makes
    ntgcalls' ffmpeg pipeline write peer audio as MP3 into a FIFO -> a child
    ffmpeg decodes the FIFO to PCM at provider.input_rate on stdout ->
    a reader task forwards chunks to the provider's send_audio.
"""
import asyncio
import audioop
import logging
import os
import time

from dotenv import load_dotenv
from pytgcalls import PyTgCalls
from pytgcalls.types import (
    ChatUpdate, Device, ExternalMedia, MediaStream, RecordStream, Update,
)
from pytgcalls.types.raw import AudioParameters
from telethon import TelegramClient

from providers import make_provider

load_dotenv()
API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
SESSION = os.environ.get("TG_SESSION", "telecall")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("telecall.bridge")

CALL_RATE = 48000
FRAME_MS = 10
FRAME_BYTES = int(CALL_RATE * 1 * 2 * FRAME_MS / 1000)  # 960 @ 48k mono 10ms

sessions: dict[int, dict] = {}


async def open_session(chat_id: int):
    provider = make_provider()
    await provider.open()
    log.info("provider=%s opened (in=%d out=%d)",
             provider.name, provider.input_rate, provider.output_rate)

    state = {
        "provider": provider,
        "up_state": None,    # provider.output_rate -> 48k
        "playback_buf": bytearray(),
        "ffmpeg_proc": None,
        "fifo_path": None,
    }
    sessions[chat_id] = state

    async def reader():
        try:
            async for ev in provider.recv():
                t = ev.get("type")
                if t == "audio":
                    pcm = ev["pcm"]
                    if provider.output_rate != CALL_RATE:
                        pcm, state["up_state"] = audioop.ratecv(
                            pcm, 2, 1, provider.output_rate, CALL_RATE, state["up_state"]
                        )
                    state["playback_buf"].extend(pcm)
                elif t == "interrupt":
                    state["playback_buf"].clear()
                elif t == "user_text":
                    log.info("USER: %s", ev.get("text", ""))
                elif t == "agent_text":
                    log.info("AGENT: %s", ev.get("text", ""))
        except Exception as e:
            log.info("provider reader exit: %s", e)

    state["reader_task"] = asyncio.create_task(reader())
    return state


def start_player(chat_id: int, calls: PyTgCalls):
    state = sessions[chat_id]
    async def player():
        period = FRAME_MS / 1000.0
        next_t = time.monotonic()
        silence = b"\x00" * FRAME_BYTES
        while chat_id in sessions:
            if len(state["playback_buf"]) >= FRAME_BYTES:
                chunk = bytes(state["playback_buf"][:FRAME_BYTES])
                del state["playback_buf"][:FRAME_BYTES]
            else:
                chunk = silence
            try:
                await calls.send_frame(chat_id, Device.MICROPHONE, chunk)
            except Exception:
                await asyncio.sleep(0.05)
                next_t = time.monotonic()
                continue
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_t = time.monotonic()
    state["player_task"] = asyncio.create_task(player())


async def start_inbound_pipe(chat_id: int, calls: PyTgCalls):
    """FIFO + ffmpeg -> provider.send_audio loop."""
    state = sessions[chat_id]
    provider = state["provider"]
    fifo_path = f"/tmp/peer_{chat_id}_{int(time.time())}.mp3"
    try: os.unlink(fifo_path)
    except FileNotFoundError: pass
    os.mkfifo(fifo_path)
    state["fifo_path"] = fifo_path

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-loglevel", "error",
        "-f", "mp3", "-i", fifo_path,
        "-f", "s16le", "-ac", "1", "-ar", str(provider.input_rate),
        "-",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    state["ffmpeg_proc"] = proc
    log.info("ffmpeg decoder spawned pid=%s (-> %d Hz)", proc.pid, provider.input_rate)

    await calls.record(
        chat_id,
        RecordStream(audio=fifo_path, audio_parameters=AudioParameters(CALL_RATE, 1)),
    )
    log.info("ntgcalls.record() started -> %s", fifo_path)

    async def reader():
        # ~100 ms chunks
        chunk_bytes = int(provider.input_rate * 2 * 0.1)
        sent = 0
        try:
            while True:
                chunk = await proc.stdout.read(chunk_bytes)
                if not chunk:
                    break
                sent += len(chunk)
                try:
                    await provider.send_audio(chunk)
                except Exception as e:
                    log.info("provider send err: %s", e); break
            log.info("inbound reader exit, %d bytes -> provider", sent)
        except asyncio.CancelledError:
            pass
    state["inbound_task"] = asyncio.create_task(reader())


async def close_session(chat_id: int):
    state = sessions.pop(chat_id, None)
    if not state:
        return
    for k in ("reader_task", "player_task", "inbound_task"):
        t = state.get(k)
        if t: t.cancel()
    proc = state.get("ffmpeg_proc")
    if proc and proc.returncode is None:
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=2)
        except Exception:
            try: proc.kill()
            except Exception: pass
    try: await state["provider"].close()
    except Exception: pass
    fifo = state.get("fifo_path")
    if fifo:
        try: os.unlink(fifo)
        except Exception: pass
    log.info("session closed chat=%s", chat_id)


async def main():
    tele = TelegramClient(SESSION, API_ID, API_HASH)
    calls = PyTgCalls(tele)

    @calls.on_update()
    async def on_update(_: PyTgCalls, update: Update):
        if isinstance(update, ChatUpdate):
            if update.status & ChatUpdate.Status.INCOMING_CALL:
                log.info("INCOMING_CALL user=%s", update.chat_id)
                try:
                    await open_session(update.chat_id)
                    await calls.play(
                        update.chat_id,
                        MediaStream(
                            media_path=ExternalMedia.AUDIO,
                            audio_parameters=AudioParameters(CALL_RATE, 1),
                        ),
                    )
                    await start_inbound_pipe(update.chat_id, calls)
                    await asyncio.sleep(0.3)
                    start_player(update.chat_id, calls)
                    log.info("call fully armed (out + in)")
                except Exception as e:
                    log.exception("accept failed: %s", e)
                    await close_session(update.chat_id)
            elif update.status & ChatUpdate.Status.DISCARDED_CALL:
                log.info("DISCARDED_CALL user=%s", update.chat_id)
                await close_session(update.chat_id)

    await calls.start()
    me = await tele.get_me()
    log.info("READY as %s (+%s) provider=%s. Call to test.",
             me.first_name, me.phone, os.environ.get("VOICE_PROVIDER", "elevenlabs"))
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
