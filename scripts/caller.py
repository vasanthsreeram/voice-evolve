"""Outbound caller — places a Telegram private call and streams a given audio file.

Usage:
    set -a && . ./.env && set +a
    TG_SESSION=telecall uv run --project telegramcaller python scripts/caller.py \
        --to-phone +6588461730 \
        --audio test_audio/test_phrase.wav \
        --hold 25

If the target user is not in the session's contacts/dialogs, --to-phone is required so
the script can import the contact (via ImportContactsRequest) before resolving the entity.
After --hold seconds the call is torn down.
"""
import argparse
import asyncio
import os

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.functions.contacts import ImportContactsRequest
from telethon.tl.types import InputPhoneContact
from pytgcalls import PyTgCalls
from pytgcalls.types import ChatUpdate, MediaStream, Update
from pytgcalls.types.raw import AudioParameters

load_dotenv()
API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
SESSION = os.environ.get("TG_SESSION", "telecall")
CALL_RATE = 48000


async def resolve_target(tele: TelegramClient, phone: str | None, user_id: int | None) -> int:
    if phone:
        try:
            res = await tele(ImportContactsRequest(contacts=[
                InputPhoneContact(client_id=0, phone=phone, first_name="callee", last_name=""),
            ]))
            if res.users:
                return res.users[0].id
        except Exception as e:
            print(f"[call] import contact failed ({e!r}); falling back to --user-id", flush=True)
    if user_id is None:
        raise SystemExit("could not resolve target — pass --user-id")
    ent = await tele.get_entity(user_id)
    return ent.id


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to-phone", help="callee phone in +E.164 form (imported as contact)")
    ap.add_argument("--user-id", type=int, help="callee Telegram user id (fallback)")
    ap.add_argument("--audio", required=True, help="audio file to stream into the call")
    ap.add_argument("--hold", type=float, default=25.0, help="seconds to keep call open")
    args = ap.parse_args()

    tele = TelegramClient(SESSION, API_ID, API_HASH)
    calls = PyTgCalls(tele)

    @calls.on_update()
    async def on_update(_, u: Update):
        if isinstance(u, ChatUpdate):
            print(f"[call] UPDATE status={u.status} chat={u.chat_id}", flush=True)

    await calls.start()
    me = await tele.get_me()
    print(f"[call] READY as {me.first_name} (+{me.phone}) session={SESSION}", flush=True)

    target = await resolve_target(tele, args.to_phone, args.user_id)
    print(f"[call] dialing {target} with {args.audio}", flush=True)
    try:
        await calls.play(
            target,
            MediaStream(
                media_path=args.audio,
                audio_parameters=AudioParameters(CALL_RATE, 1),
            ),
        )
        print("[call] play() returned — call up, streaming audio", flush=True)
    except Exception as e:
        print(f"[call] play() error: {e!r}", flush=True)
        return

    await asyncio.sleep(args.hold)
    print("[call] hanging up", flush=True)
    try:
        await calls.leave_call(target)
    except Exception as e:
        print(f"[call] leave_call err: {e!r}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
