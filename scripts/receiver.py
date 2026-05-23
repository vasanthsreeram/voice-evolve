"""Auto-accepts an inbound Telegram private call and records the caller's audio.

Usage:
    set -a && . ./.env && set +a
    TG_SESSION=bob uv run --project telegramcaller python scripts/receiver.py

Writes one WAV per accepted call to recordings/inbound_<ts>_<chat_id>.wav
(mono, 48 kHz). Plays silence back so the call stays connected for the caller.
"""
import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from pytgcalls import PyTgCalls
from pytgcalls.types import (
    ChatUpdate, Device, ExternalMedia, MediaStream, RecordStream, Update,
)
from pytgcalls.types.raw import AudioParameters

load_dotenv()
API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
SESSION = os.environ.get("TG_SESSION", "bob")
REC_DIR = Path(__file__).resolve().parent.parent / "recordings"
CALL_RATE = 48000
FRAME_MS = 10
FRAME_BYTES = int(CALL_RATE * 2 * FRAME_MS / 1000)

state: dict[int, dict] = {}


async def silence_player(calls: PyTgCalls, chat_id: int):
    silence = b"\x00" * FRAME_BYTES
    period = FRAME_MS / 1000.0
    nxt = time.monotonic()
    while chat_id in state:
        try:
            await calls.send_frame(chat_id, Device.MICROPHONE, silence)
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


async def start_recording(calls: PyTgCalls, chat_id: int):
    REC_DIR.mkdir(parents=True, exist_ok=True)
    fifo = f"/tmp/peer_{chat_id}_{int(time.time())}.mp3"
    try:
        os.unlink(fifo)
    except FileNotFoundError:
        pass
    os.mkfifo(fifo)
    wav = REC_DIR / f"inbound_{int(time.time())}_{chat_id}.wav"
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "mp3", "-i", fifo,
        "-ac", "1", "-ar", str(CALL_RATE), str(wav),
    )
    await calls.record(
        chat_id,
        RecordStream(audio=fifo, audio_parameters=AudioParameters(CALL_RATE, 1)),
    )
    state[chat_id] = {"fifo": fifo, "proc": proc, "wav": wav}
    await asyncio.sleep(0.3)
    state[chat_id]["player"] = asyncio.create_task(silence_player(calls, chat_id))
    print(f"[recv] recording -> {wav}", flush=True)


async def stop_recording(chat_id: int):
    s = state.pop(chat_id, None)
    if not s:
        return
    t = s.get("player")
    if t:
        t.cancel()
    p = s.get("proc")
    if p and p.returncode is None:
        try:
            p.terminate()
            await asyncio.wait_for(p.wait(), 5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    try:
        os.unlink(s["fifo"])
    except Exception:
        pass
    print(f"[recv] saved {s['wav']}", flush=True)


async def main():
    tele = TelegramClient(SESSION, API_ID, API_HASH)
    calls = PyTgCalls(tele)

    @calls.on_update()
    async def on_update(_, u: Update):
        if not isinstance(u, ChatUpdate):
            return
        if u.status & ChatUpdate.Status.INCOMING_CALL:
            cid = u.chat_id
            print(f"[recv] INCOMING_CALL from {cid} — accepting", flush=True)
            try:
                await calls.play(cid, MediaStream(
                    media_path=ExternalMedia.AUDIO,
                    audio_parameters=AudioParameters(CALL_RATE, 1),
                ))
                await start_recording(calls, cid)
                print("[recv] armed", flush=True)
            except Exception as e:
                print(f"[recv] accept error: {e!r}", flush=True)
                await stop_recording(cid)
        elif u.status & ChatUpdate.Status.DISCARDED_CALL:
            print(f"[recv] DISCARDED_CALL {u.chat_id}", flush=True)
            await stop_recording(u.chat_id)

    await calls.start()
    me = await tele.get_me()
    print(f"[recv] READY as {me.first_name} (+{me.phone}) session={SESSION}", flush=True)
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
